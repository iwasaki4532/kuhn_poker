import streamlit as st   # Webアプリを作るためのフレームワーク
import random             # シャッフルや乱数（先手後手の決定など）に使う
import numpy as np        # 戦略やベクトル演算（配列計算）に使う
import pickle             # 学習結果（node_map）をファイルに保存/読込する
import os                 # ファイルパスの操作・存在チェックに使う
import itertools          # J/Q/Kの並べ替え（順列）を生成するために使う

# ============================================================
# 定数定義
# ============================================================


CARD_VALUES = {'J': 1, 'Q': 2, 'K': 3}  # カードの強さを数値化した辞書（J<Q<K）



STARTING_CHIPS = 20  # 開始時のチップ枚数

# ============================================================
# ゲームのルール（終端判定）
# ============================================================


# 学習時はこの6通りを順番に巡回することで、ランダムシャッフルのような
# サンプリングのばらつきなく、すべての配り方を厳密に均等に学習できる。
DECK_PERMS = list(itertools.permutations(('J', 'Q', 'K')))


def terminal_util(cards, history):
    """
    ゲームが終端状態にあるとき「現在プレイヤー視点の利得」を返す。
    まだゲームが続いている場合は None を返す。

    【現在プレイヤーとは】
    履歴の長さ plays の偶奇でターンが決まる。
        plays % 2 == 0 → P0 のターン（先手）
        plays % 2 == 1 → P1 のターン（後手）
    終端ノードでは「次に行動するはずだったプレイヤー」が current player。

    Args:
        cards  : [P0のカード文字列, P1のカード文字列]
        history: アクション履歴文字列

    Returns:
        利得（float）または None（未終了）
    """
    plays = len(history)        # これまでに行われたアクションの数
    player = plays % 2          # 0ならP0のターン、1ならP1のターン（次に行動する側）
    opponent = 1 - player        # playerの逆（0なら1、1なら0）

    if plays > 1:               
        if history == 'pp':
            # 両者パス → カードの大小だけで勝敗を決める
            return 1.0 if CARD_VALUES[cards[player]] > CARD_VALUES[cards[opponent]] else -1.0

        if history == 'bp' or history == 'pbp':
            # 相手がフォールドしたので無条件に +1
            return 1.0

        if history == 'bb' or history == 'pbb':
            # 両者ベット（コール含む）→ カードの大小で±2を決める
            return 2.0 if CARD_VALUES[cards[player]] > CARD_VALUES[cards[opponent]] else -2.0

    return None


# ============================================================
# CFRノード
# ============================================================

class KuhnNode:
    """
    - regret_sum  : 各アクションの累積後悔量。
                    →もしあのアクションを取っていたら得られた利得と
                    実際に得た利得の差を反復ごとに積み上げたもの。
    - strategy_sum: 学習全体を通じた平均的な戦略を記録
    """

    def __init__(self, n_actions=2):
      
        self.regret_sum = np.zeros(n_actions)    # 累積後悔量 [pの後悔, bの後悔]
        self.strategy_sum = np.zeros(n_actions)  # 戦略の累積値

    def get_strategy(self, realization_weight):
        """
        現在の累積後悔量から「今回の反復で使う戦略」を計算する。
        Args:
            realization_weight: このノードへ到達するまでの確率（0〜1）。
                                 そのプレイヤー自身が関与する部分だけを掛け合わせた値。

        Returns:
            strategy: 各アクションの確率を格納した numpy 配列（合計=1）。
        """
        strategy = np.maximum(self.regret_sum, 0)   # 負の後悔を0にする（マイナス部分を切り捨て）
        normalizing_sum = np.sum(strategy)           # 正の後悔の合計値
        if normalizing_sum > 0:
            strategy /= normalizing_sum              # 合計が1になるように正規化（確率分布化）
        else:
            # 学習初期の場合は均等確率にす
            strategy = np.repeat(1.0 / len(strategy), len(strategy))

        self.strategy_sum += realization_weight * strategy  
        return strategy

    def get_average_strategy(self):
        """
        学習全体を通じた「平均戦略」を返す。

        Returns:
            平均戦略の確率分布 numpy 配列（合計=1）。
        """
        normalizing_sum = np.sum(self.strategy_sum)  # 積算してきた戦略の合計
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum  # 合計1になるよう正規化して返す
        else:
            # 一度もこのノードに到達していない場合は均等確率を返す
            return np.repeat(1.0 / len(self.strategy_sum), len(self.strategy_sum))


# ============================================================
# CFR学習器
# ============================================================

class KuhnCFRTrainer:
     # クーンポーカー全体の CFR 学習を管理するクラス。


    def __init__(self):
        self.node_map = {}  # 情報集合キー("カード:履歴") → KuhnNode のマップ

    def cfr(self, cards, history, p0, p1):
        """
        CFR の中核となる再帰関数。ゲーム木を深さ優先で探索しながら
        各情報集合の後悔量と戦略を更新する。

        Args:
            cards  : [P0のカード, P1のカード]
            history: これまでのアクション履歴文字列
            p0     : P0 が自分のアクション選択によってここへ到達した確率
            p1     : P1 が自分のアクション選択によってここへ到達した確率

        Returns:
            current player 視点の期待利得（float）
        """
        plays = len(history)
        player = plays % 2  

        util = terminal_util(cards, history)
        if util is not None:
            return util  # 終端に達していれば、その利得をそのまま返して終了する

        # 「自分のカード:これまでの履歴」を情報集合キーにする。
        # 例: "Q:pb" → Qを持ち、先手がパス・後手がベットした状況。
        info_set = cards[player] + ":" + history
        if info_set not in self.node_map:
            self.node_map[info_set] = KuhnNode()  # 初めて訪れる情報集合ならノードを新規作成

        node = self.node_map[info_set]

        # 戦略の重み = 自分がここへ到達した確率（相手の行動は含めない）
        realization_weight = p0 if player == 0 else p1
        strategy = node.get_strategy(realization_weight)  # 今回の反復で使う確率分布[p,b]

        # 各アクション（0=p, 1=b）について子ノードの期待利得を再帰計算する。
        # 再帰で返ってくる値は「次プレイヤー視点」なので符号を反転して
        # 「現プレイヤー視点」の利得に変換する（ゼロサムゲームの性質を利用）。
        action_utils = np.zeros(2)  # 各アクションを取った場合の利得を格納する配列
        node_util = 0                # このノードの（戦略に従った場合の）期待利得
        actions = ['p', 'b']

        for a in range(2):
            next_history = history + actions[a]  # 履歴にアクション文字を追加して子ノードへ
            if player == 0:
                # P0 がアクション a を選んだ → P0 の到達確率に strategy[a] を掛ける
                action_utils[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                # P1 がアクション a を選んだ → P1 の到達確率に strategy[a] を掛ける
                action_utils[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * action_utils[a]  # 期待利得 = Σ(確率×利得)

        # 後悔量を更新：「そのアクションを選んでいた場合の利得」-「実際の期待利得」
        # opponent_weight を掛けることで「相手がここに来やすいほど後悔を重視」する。
        opponent_weight = p1 if player == 0 else p0  # 相手側の到達確率
        for a in range(2):
            regret = action_utils[a] - node_util       # そのアクションを取っていれば得られた差分
            node.regret_sum[a] += opponent_weight * regret  # 重み付けして累積後悔に加算

        return node_util  # このノードの（現在プレイヤー視点の）期待利得を呼び出し元に返す

    def train(self, iterations, on_progress=None):
        """
        指定した反復回数だけ CFR を実行して戦略を学習する。

        Args:
            iterations : 学習の反復回数（= ゲーム木を探索する cfr 呼び出し回数）。
            on_progress: 進捗通知コールバック。毎反復 on_progress(done, total) の
                         形で呼ばれる（None なら何もしない）。UI の進捗バー更新に使う。
        """
        n_deals = len(DECK_PERMS)  # 配り方の総数（=6）
        for i in range(iterations):
            self.cfr(DECK_PERMS[i % n_deals], "", 1.0, 1.0)  # i%6 で6通りを順番に使う。履歴は空文字で開始
            if on_progress is not None:
                on_progress(i + 1, iterations)  # 進捗（現在の反復数, 総反復数）を通知

    def save_model(self, filename):
        """
        学習済み node_map を pickle 形式でファイルに保存する。

        pickle はオブジェクトをバイト列にシリアライズする Python 標準の仕組み。
        信頼できる自分自身が作ったファイルを読み書きするだけなので安全に使える。

        Args:
            filename: 保存先ファイルパス文字列
        """
        with open(filename, 'wb') as f:    # バイナリ書き込みモードでファイルを開く
            pickle.dump(self.node_map, f)  # node_map（辞書）をそのままバイト列として保存

    def load_model(self, filename):
        """
        ファイルから node_map を読み込む。

        ファイルが存在しない場合（初回起動時など）は False を返し、
        呼び出し元で学習→保存を行わせる設計にしている。

        Args:
            filename: 読み込むファイルパス文字列

        Returns:
            ロードに成功したら True、ファイルが存在しなければ False
        """
        if os.path.exists(filename):        # ファイルの有無を先にチェック
            with open(filename, 'rb') as f:  # バイナリ読み込みモードで開く
                self.node_map = pickle.load(f)  # 保存時と同じ構造の辞書として復元
            return True
        return False  # ファイルが無い＝未学習。呼び出し元で学習させる


# ============================================================
# モデルのロード（Streamlit キャッシュで初回のみ実行）
# ============================================================

TRAIN_ITERATIONS = 200_000


@st.cache_resource(show_spinner="🤖 AIを学習中... 初回のみ約30秒かかります")
def load_trainer():
    """
    @st.cache_resource によりアプリ起動後の初回のみ学習を実行し、
    結果をメモリにキャッシュする。ファイルへの保存は行わない。
    """
    trainer = KuhnCFRTrainer()
    trainer.train(TRAIN_ITERATIONS)
    return trainer


# ============================================================
# セッション状態の初期化
# ============================================================

def init_session():
    """
    st.session_state にゲームの初期値を設定する。

    【st.session_state とは】
    Streamlit はユーザー操作（ボタン押下など）のたびにスクリプト全体を
    再実行するため、通常の Python 変数は毎回リセットされてしまう。
    st.session_state はブラウザセッション単位で値を永続化する辞書で、
    ゲームの状態管理に必須の仕組みとなる。

    【各状態変数の役割】
    player_chips     : 人間プレイヤーの現在のチップ残高
    round_count      : これまでに行ったラウンド数（表示用）
    phase            : 画面の状態機械の現在フェーズ
                         'start'     → ゲーム開始前の案内画面
                         'playing'   → プレイヤーが行動を選択する画面
                         'round_end' → ラウンド結果を表示する画面
                         'game_over' → ゲーム終了後の成績画面
    cards            : 配られたカード [P0のカード文字列, P1のカード文字列]
    history          : その局のアクション履歴（例: "pb" = パス→ベット）
    human_goes_first : True なら人間が先手(P0)、False なら後手(P1)
    current_player   : 現在アクションすべきプレイヤー番号（0=先手P0 / 1=後手P1）
    last_delta       : 直前ラウンドの人間の収支（+2/-2/+1/-1）
    round_log        : 画面表示用のアクション履歴テキストリスト

    
    *streamlit のスクリプト再実行ごとにこの関数が呼ばれるが、
    if k not in st.session_state のガードで既存の値を守ることで
    ゲーム中の状態がリセットされるのを防ぐ。
    """
    defaults = {
        'player_chips': STARTING_CHIPS,  
        'round_count': 0,                
        'phase': 'start',                
        'cards': None,                   
        'history': '',                   
        'human_goes_first': True,        # 暫定値（start_new_roundで毎回上書きされる）
        'current_player': 0,             # 常にP0（先手）から開始
        'last_delta': 0,                 # 直前の収支
        'round_log': [],                 # ログは空リストから開始
    }
    for k, v in defaults.items():
        if k not in st.session_state:    # まだ存在しないキーだけ初期値を設定（既存値は保持）
            st.session_state[k] = v


# ============================================================
# ラウンド開始
# ============================================================

def start_new_round():
    """
    新しいラウンドの準備を行い、フェーズを 'playing' に移行する。

    シャッフル後、deck[0] → P0（先手）、deck[1] → P1（後手）と固定する。
    人間がどちらの役割でも、ゲームロジックの中では常に P0=cards[0]、
    P1=cards[1] で処理し、表示や報酬計算のときだけ human_goes_first
    フラグで「人間がP0かP1か」を補正する設計にしている。
    """
    deck = ['J', 'Q', 'K']
    random.shuffle(deck)  # J/Q/Kの順序をランダムに入れ替える（3枚中2枚だけ使う）
    st.session_state.cards = [deck[0], deck[1]]  # deck[0]→P0, deck[1]→P1（deck[2]は未使用）
    st.session_state.human_goes_first = random.choice([True, False])  # 先手後手をランダム決定
    st.session_state.current_player = 0  # 必ずP0（先手）から行動を開始
    st.session_state.round_count += 1    # ラウンド数を1増やす
    st.session_state.phase = 'playing'   # フェーズをプレイ中に変更
    st.session_state.round_log = []      # 前ラウンドのログをクリア
    st.session_state.history = ''        # 履歴をクリア


# ============================================================
# プレイヤーのインデックス補正
# ============================================================

def human_card_index():
    """
    人間プレイヤーのカードが cards 配列のどちらに入っているかを返す。

    人間が先手(P0)なら cards[0]、後手(P1)なら cards[1]。
    表示や勝敗計算で「人間のカード／AIのカード」を正しく取り出すために使う。

    Returns:
        人間のカードのインデックス（0 or 1）
    """
    return 0 if st.session_state.human_goes_first else 1  # 人間が先手ならcards[0]、後手ならcards[1]


def is_human_turn():
    """
    現在のターンが人間のターンかどうかを返す。

    【P0/P1 と先手/後手の対応】
    ゲームロジック内では P0 が常に先手（current_player=0 のとき行動）、
    P1 が常に後手（current_player=1 のとき行動）として扱う。

    人間の役割と current_player の組み合わせで判断する：
      human_goes_first=True  かつ current_player=0 → 人間のターン（人間が先手=P0）
      human_goes_first=False かつ current_player=1 → 人間のターン（人間が後手=P1）
      その他 → AI のターン

    Returns:
        人間のターンなら True、AI のターンなら False
    """
    cp = st.session_state.current_player    # 現在行動すべきプレイヤー番号（0 or 1）
    hgf = st.session_state.human_goes_first  # 人間が先手かどうか
    return (cp == 0 and hgf) or (cp == 1 and not hgf)  # 「人間の役割」と「現在の手番」が一致するか


# ============================================================
# 報酬計算
# ============================================================

def compute_p0_delta(cards, history):
    """
    ラウンド終了時の P0（先手）視点の収支チップ数を返す。
    まだゲームが終わっていなければ None を返す。

    Args:
        cards  : [P0のカード, P1のカード]
        history: アクション履歴文字列

    Returns:
        P0 視点の収支チップ数（int）またはゲーム未終了を示す None
    """
    util = terminal_util(cards, history)
    if util is None:
        return None  # まだゲームが終わっていない

    # terminal_util は「現在プレイヤー視点」の利得を返す。終端履歴が偶数長なら
    # 現在プレイヤー=P0、奇数長なら=P1 なので、P1視点（pbp / pbb）のときだけ
    # 符号を反転して「P0視点の収支」に統一する。
    return int(util) if len(history) % 2 == 0 else int(-util)


def compute_delta(cards, history, human_goes_first):
    """
    ラウンド終了時の人間プレイヤーの収支チップ数を返す。
    まだゲームが終わっていなければ None を返す。

    【P0 視点から人間視点への変換】
    ゲームロジックは「P0 が勝ったか負けたか」で計算する（compute_p0_delta）。
    人間が先手(P0)なら符号そのまま、後手(P1)なら符号を反転して返す。

    Args:
        cards            : [P0のカード, P1のカード]
        history          : アクション履歴文字列
        human_goes_first : True=人間が先手(P0)、False=人間が後手(P1)

    Returns:
        人間の収支チップ数（int）またはゲーム未終了を示す None
    """
    p0_delta = compute_p0_delta(cards, history)
    if p0_delta is None:
        return None  # まだゲームが終わっていない
    return p0_delta if human_goes_first else -p0_delta  # 人間が後手なら符号を反転


# ============================================================
# アクション適用
# ============================================================

def apply_action(action, is_human):
    """
    プレイヤーのアクションをゲーム状態に反映する。

    【処理の流れ】
    1. ログにアクション内容を追記（画面表示用）
    2. 履歴文字列にアクション文字（'p' または 'b'）を追加
    3. current_player を 0→1 または 1→0 に切り替え
    4. ゲーム終了判定を行い、終了していればチップを増減してフェーズ移行

    Args:
        action   : 'p'（パス/フォールド）または 'b'（ベット/コール）
        is_human : 人間が行動したなら True、AI なら False（ログ表示用）
    """
    actor = "あなた" if is_human else "AI"
    label = "Bet / Call" if action == 'b' else "Pass / Fold"
    st.session_state.round_log.append(f"{actor}: {label}")  # 画面表示用のログに追記

    st.session_state.history += action               # 履歴文字列にアクション文字を追加
    st.session_state.current_player = 1 - st.session_state.current_player  # 手番を相手に渡す（0⇔1）

    delta = compute_delta(
        st.session_state.cards,
        st.session_state.history,
        st.session_state.human_goes_first,
    )
    if delta is not None:                       # ゲームが終了した場合のみ
        st.session_state.last_delta = delta            # 直前ラウンドの収支を記録
        st.session_state.player_chips += delta         # チップ残高に反映
        st.session_state.phase = 'round_end'           # 結果表示フェーズへ移行


# ============================================================
# AI のアクション選択
# ============================================================

def get_ai_action(trainer, card, history):
    """
    学習済みの平均戦略に基づいて AI のアクションを確率的に選択する。

    【情報集合キーの組み立て】
    AI は自分のカードとこれまでの履歴だけを知ることができる。
    "カード文字:履歴文字列" を情報集合のキーとして node_map を引く。
    例: AI がQを持ち、相手がパスした状況 → "Q:p"

    【np.random.choice による確率的選択】
    平均戦略は各アクションの確率を表す配列（例: [0.33, 0.67]）。
    np.random.choice(['p', 'b'], p=strategy) でその確率に従って
    ランダムにアクションを1つ選ぶ。

    Args:
        trainer: 学習済み KuhnCFRTrainer インスタンス
        card   : AI のカード文字列（'J', 'Q', 'K'）
        history: これまでのアクション履歴文字列

    Returns:
        'p' または 'b' のアクション文字列
    """
    info_set = card + ":" + history  # 例: "Q:p"
    if info_set not in trainer.node_map:
        return 'p'  
    strategy = trainer.node_map[info_set].get_average_strategy()  # [pの確率, bの確率]
    return np.random.choice(['p', 'b'], p=strategy)  # 確率に従ってアクションを抽選


# ============================================================
# ゲームリセット
# ============================================================

def reset_game():
  #session_state をすべて削除してゲームを完全に初期状態に戻す。

    for k in list(st.session_state.keys()):  # キー一覧を先にコピー
        del st.session_state[k]               # すべてのセッション変数を削除


# ============================================================
# 各フェーズの描画関数
# ============================================================

def render_playing(trainer):
    """
    'playing' フェーズの描画と AI の自動進行を担う。

    【AI ターンの自動処理ループ】
    Streamlit は「ボタンを押す」などのイベントがないと自動では再描画しない。
    そこで AI のターンのとき：
      1. AI のアクションを計算して apply_action() で状態を更新
      2. st.rerun() でスクリプトを強制再実行（= 再描画）
      3. 再描画後も AI ターンなら再び同じ処理 → 人間ターンになるまで繰り返す
    という仕組みで「AI が連続してアクションする」状況に対応している。

    【人間のカードとポジション表示】
    人間が先手(P0)なら cards[0]、後手(P1)なら cards[1] を取り出して表示する。
    AI はその逆のインデックスのカードで自分の戦略を決める。

    Args:
        trainer: 学習済み KuhnCFRTrainer インスタンス（AI の行動選択に使用）
    """
    human_goes_first = st.session_state.human_goes_first
    h_idx = human_card_index()      # 人間のカードのインデックス
    ai_idx = 1 - h_idx              # AI のカードのインデックス

    if not is_human_turn():
        # AI は自分（P0 か P1）のカードを使って戦略を決める
        ai_card = st.session_state.cards[ai_idx]
        action = get_ai_action(trainer, ai_card, st.session_state.history)  # 戦略に従ってp/bを抽選
        apply_action(action, is_human=False)  # ゲーム状態に反映
        st.rerun()  # 再描画 → 再度この関数が呼ばれ、人間ターンになるまでAIが行動を続ける
        return

    cards = st.session_state.cards

    col1, col2 = st.columns(2)  # 画面を2列に分割
    with col1:
        st.metric("あなたのカード", cards[h_idx])  # 自分のカードのみ表示（相手のカードは非公開）
    with col2:
        st.metric("役割", "先手" if human_goes_first else "後手")

    if st.session_state.round_log:
        st.write("**アクション履歴**")
        for entry in st.session_state.round_log:  # これまでのアクションを順に表示
            st.write(f"　{entry}")

    st.divider()
    st.write("**アクションを選択してください:**")
    col_p, col_b = st.columns(2)
    with col_p:
        if st.button("Pass / Fold　(p)", use_container_width=True):
            apply_action('p', is_human=True)  # パス/フォールドを適用
            st.rerun()                         # 再描画して結果を反映
    with col_b:
        if st.button("Bet / Call　(b)", use_container_width=True, type="primary"):
            apply_action('b', is_human=True)  # ベット/コールを適用
            st.rerun()


def render_round_end():
    """
    'round_end' フェーズの描画。両者のカードを公開して結果を示す。

    【カードの公開】
    playing フェーズでは AI のカードは非公開だが、
    ラウンド終了後は AI のカードも表示して結果の透明性を確保する。
    人間／AI のカードは human_goes_first に応じてインデックスを補正して取り出す。

    【チップ切れ判定】
    player_chips が 0 以下になった場合はゲームオーバーボタンだけを表示し、
    通常時は「次のラウンドへ」「ゲームを終了する」の2択を提示する。
    """
    cards = st.session_state.cards
    delta = st.session_state.last_delta
    history = st.session_state.history
    chips = st.session_state.player_chips

    h_idx = human_card_index()
    ai_idx = 1 - h_idx

    st.write(f"**アクション履歴:** `{history}`")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("あなたのカード", cards[h_idx])
    with col2:
        st.metric("AIのカード", cards[ai_idx])  # ラウンド終了後はAIのカードも公開する

    if delta > 0:
        st.success(f"あなたの勝ち！　+{delta} チップ")
    elif delta < 0:
        st.error(f"あなたの負け…　{delta} チップ")
    else:
        st.info("引き分け")  

    st.divider()

    if chips <= 0:
        st.error("チップがなくなりました。ゲームオーバー！")
        if st.button("最初からやり直す", type="primary"):
            reset_game()  # session_stateを全削除
            st.rerun()    # → init_sessionで初期値が再設定され、startフェーズに戻る
    else:
        col_next, col_quit = st.columns(2)
        with col_next:
            if st.button("次のラウンドへ", use_container_width=True, type="primary"):
                start_new_round()  # 新しいカード配布・先手後手抽選を行いplayingへ
                st.rerun()
        with col_quit:
            if st.button("ゲームを終了する", use_container_width=True):
                st.session_state.phase = 'game_over'  # 成績画面フェーズへ
                st.rerun()


def render_game_over():
    """
    'game_over' フェーズの描画。ゲーム全体を通じた成績を表示する。

  
    最終持ち金 : 現在保持しているチップ総数
    収支       : STARTING_CHIPS との差分（プラスなら黒字、マイナスなら赤字）
    総ラウンド数: 何ラウンド遊んだか（1ラウンド当たりの収支を自分で計算できる）
    """
    chips = st.session_state.player_chips
    rounds = st.session_state.round_count
    diff = chips - STARTING_CHIPS  # 開始時チップとの差分（プラスなら黒字、マイナスなら赤字）

    col1, col2, col3 = st.columns(3)  # 3つの指標を横並びで表示
    with col1:
        st.metric("最終持ち金", f"{chips} チップ")
    with col2:
        st.metric("収支", diff)
    with col3:
        st.metric("総ラウンド数", f"{rounds} R")

    st.divider()
    if st.button("もう一度プレイ", type="primary"):
        reset_game()  # 全状態を削除
        st.rerun()    # startフェーズからやり直す


# ============================================================
# メイン関数
# ============================================================

def main():
    """

    【フェーズ遷移の状態機械】
    start → playing → round_end ─→ playing（次ラウンド）
                              └→ game_over（終了選択 or チップ切れ）
    game_over → start（リセット後）
    """
    st.set_page_config(page_title="クーン ポーカー", page_icon="🃏", layout="centered")  # ページの基本設定
    st.title("🃏 クーン ポーカー")
    st.caption("J < Q < K の3枚を使ったポーカー。AIはCFRアルゴリズムで学習済みです。")

    init_session()        # session_stateの初期化（既存値は維持）
    trainer = load_trainer()  # キャッシュされた学習済みtrainerを取得（初回のみ学習）

    with st.sidebar:
        st.header("ゲーム情報")
        chips = st.session_state.player_chips
        diff = chips - STARTING_CHIPS
        st.metric(
            "持ち金",
            f"{chips} チップ",
            delta=diff if st.session_state.round_count > 0 else None,
        )
        st.metric("ラウンド数", st.session_state.round_count)
        st.divider()
        st.write("**ルール**")
        st.write("カード強さ: J < Q < K")
        st.write("p = Pass / Fold")
        st.write("b = Bet / Call")
        st.write("先手がパスして後手がベットして先手がパス (pbp) → 後手の勝ち")
        st.write("両者ベット (bb / pbb) → 強いカードが勝ち (+2)")

    phase = st.session_state.phase  # 現在のフェーズを取得し、対応する描画処理に分岐

    if phase == 'start':
        st.info(f"開始チップ: **{STARTING_CHIPS} チップ**　チップが0になるとゲームオーバーです。")
        if st.button("ゲーム開始", type="primary", use_container_width=True):
            start_new_round()  # 最初のラウンドを準備
            st.rerun()

    elif phase == 'playing':
        st.subheader(f"ラウンド {st.session_state.round_count}")
        render_playing(trainer)

    elif phase == 'round_end':
        st.subheader(f"ラウンド {st.session_state.round_count}　結果")
        render_round_end()

    elif phase == 'game_over':
        st.subheader("ゲーム終了")
        render_game_over()


if __name__ == "__main__":
    main()  # スクリプトが直接実行されたときのみmain()を呼ぶ（streamlit runからの実行に対応）
