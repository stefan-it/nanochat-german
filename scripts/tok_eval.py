# Work is based on the original implementation by Andrej Karpathy, licensed under MIT.
# Snapshot: https://github.com/karpathy/nanochat/blob/d6d86cbf4c0bcc1de5bbab28cae1f98038d0362a/scripts/tok_eval.py

"""
Evaluate compression ratio of the tokenizer.
"""

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# Source: https://www.tagesschau.de/ausland/europa/louvre-einbruch-102.html
news_text = r"""
So lief der Einbruch in den Louvre ab
Stand: 19.10.2025 22:33 Uhr

Es ist eine Tat, die Frankreich aufwühlt: Nur vier Minuten brauchten vermummte Diebe, um in das weltberühmte Museum Louvre einzubrechen und Teile der Kronjuwelen zu erbeuten. Was über den Coup bekannt ist.

Wie kamen die Täter in das Gebäude?

Die vollständig vermummten Täter trafen nach ersten Erkenntnissen gegen 9.30 Uhr, kurz nach Öffnung des Louvre, am Gebäude ein. Auf der zur Seine gewandten Seite des Museums parkten sie ein Fahrzeug mit Hebebühne. Es handelt sich um einen herkömmlichen Möbellift, wie er auch bei Umzügen verwendet wird.

Über diese Hebebühne gelangten die Täter auf einen etwa zehn Meter hohen Balkon. Mehrere Medien berichten, zwei Männer seien in das Innere des Louvre eingedrungen, nachdem sie die Fenster mit einem Trennschleifer oder kleinen Kettensägen 
zerstört hätten. Ein dritter Mann habe draußen Wache gestanden.

Wie lief der Diebstahl ab?

Laut Frankreichs Kulturministerin Rachida Dati benötigten die Täter für ihren Beutezug nur vier Minuten. Durch das Fenster gelangten sie in den Denon-Flügel des Louvre, in dem auch die "Mona Lisa", das weltberühmte Gemälde von Leonardo da Vinci, ausgestellt wird. Die Täter standen direkt in der Apollon-Galerie, in der die französischen Kronjuwelen aufbewahrt werden. Dort befinden sich Schmuckstücke aus der Zeit Ludwigs des Vierzehnten und der napoleonischen Kaiserzeit, darunter drei große Diamanten.

Dati zufolge wurden Vitrinen zerstört und Wertstücke entnommen. Innenminister Laurent Nuñez sagte: "Die Einbrecher hatten den Ort vorher offensichtlich genau erkundet und sie waren sehr versiert." Nach dem Diebstahl flüchteten sie nach Informationen der Zeitung Le Parisien auf zwei hochmotorisierten Motorrollern. Die Videoüberwachung habe sie auf dem Weg in Richtung der Autobahn A6 gefilmt.
""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

# Taken from: https://git.informatik.uni-leipzig.de/fl34gufe/skripte
math_text = r"""
\documentclass[12pt]{article}

\maketitle

\nocite{*}
\bibliography{literatur}
\bibliographystyle{abbrv}

\section*{Grundlagen}

Mathematische Objekte bestehen aus Grundmengen ggf. Relationen, Funktionen und Konstanten.

Einfache Aussagen betreffen nur Elemente der Grundmenge und haben keine unendlichen Dis- oder Konjunktionen.
Dies sind prädikatenlogische Aussagen erster Stufe.

\textit{Monadische Aussagen 2. Stufe erlaubten Quantifizierung über Teilmengen der Grundmenge; Aussagen 2. Stufe erlaubten zusätzlich Quantifizierung über Funktionen und Relationen.}

\section{Strukturen}

\begin{dfn}
    Eine \underline{Signatur} $ \sigma $ ist ein Quadrupel
    \begin{equation*}
        \sigma = \sign{}
    \end{equation*}
    mit einer Menge von Konstantensymbolen $ \m{C} $, einer Menge von Funktionssymbolen $ \m{F} $, einer Menge von Relationensymbolen $ \m{R} $, einer Stelligkeitsfunktion $ \sigma' : \m{F} \cup \m{R} \rightarrow \mathbb{N} $.
\end{dfn}

\begin{dfn}
    Eine \underline{Struktur} $ \m{M} $ ist ein Quadrupel
    \begin{equation*}
        \m{M} = \struc{M}{\m{M}}{}
    \end{equation*}
    mit einer Menge $ M $ (oftmals $ M \neq \emptyset $), Indexmengen $ \m{C}, \m{F}, \m{R} $.
    Wobei gilt $ c^\m{M} \in M $ für $ c \in \m{C} $, $ f^\m{M} : M^{n_f} \rightarrow M $, $ n_f \in \mathbb{N} $ für $ f \in \m{F} $, $ R^\m{M} \subseteq M^{m_R} $ für $ R \in \m{R} $.

    $ \m{M} $ heißt \underline{$ \sigma $-Struktur} bzw. $ \m{M} $ und $ \sigma $ \underline{passen zueinander}, falls:
    \begin{enumerate}
        \item $ n_f = \sigma'(f) $ für $ f \in \m{F} $
        \item $ m_R = \sigma'(R) $ für $ R \in \m{R} $
    \end{enumerate}

    $ c^\m{M} $ heißt auch \underline{Interpretation} von $ c $ in $ \m{M} $; Analoges gilt für $ f^\m{M}, R^\m{M} $.
\end{dfn}

\begin{dfn}
    Seien $ \m{M} = \struc{M}{\m{M}}{} $, $ \m{N} = \struc{N}{\m{N}}{} $ zwei $ \sigma $-Strukturen.
    Ein Homomorphismus von $ \m{M} $ nach $ \m{N} $ ist eine Abbildung $ h : M \rightarrow N $ mit:
    \begin{enumerate}
        \item $ h(c^\m{M}) = c^\m{N} $ für allen Konstantensymbole $ c \in \m{C} $
        \item $ h\big(f^\m{M}(a_1, ..., a_n)\big) = f^\m{N}\big(h(a_1), ..., h(a_n)\big) $ für alle Funktionssymbole $ f \in \m{F} $ mit $ n = \sigma'(f) $, $ a_1, ..., a_n \in M $.
        \item \label{itm:homomorphismus-3} $ (a_1, ..., a_n) \in R^\m{M} \Rightarrow \big(h(a_1), ..., h(a_n)\big) \in R^\m{N} $ für alle Relationensymbole $ R \in \m{R} $, $ n = \sigma'(R) $, $ a_1, ..., a_n \in M $.
    \end{enumerate}

        Für einen Homomorphismus $ h $ von $ \m{M} $ nach $ \m{N} $ schreibt man auch $ h : \m{M} \rightarrow \m{N} $.

        $ h $ heißt \underline{stark}, wenn für alle $ R \in \m{R} $ mit $ n = \sigma'(R) $, $ b_1, ..., b_n \in h(M) \subseteq N $ mit $ (b_1, ..., b_n) \in R^\m{N} $ gilt:
        \begin{equation}
            \label{eq:staerke}
            \exists a_1, ..., a_n \in M : (a_1, ..., a_n) \in R^\m{M}, h(a_i) = b_i, 1 \leq i \leq n
        \end{equation}

        $ h $ heißt \underline{Einbettung}, falls $ h $ stark und injektiv ist.
        Ist $ h $ injektiv, dann lassen sich Bedingung \ref{itm:homomorphismus-3} und Gleichung \eqref{eq:staerke} folgendermaßen zusammenfassen; für alle $ R \in \m{R} $, $ n = \sigma'(R) $, $ a_1, ..., a_n \in M $:
        \begin{equation*}
            (a_1, ..., a_n) \in R^\m{M} \Leftrightarrow \big(h(a_1), ..., h(a_n)\big) \in R^\m{N}
        \end{equation*}

        $ h $ heißt Isomorphimus, wenn $ h $ eine surjektive Einbettung ist.
        Man schreibt $ \m{M} \cong \m{N} $, wenn ein Isomorphimus $ h : \m{M} \rightarrow \m{N} $ existiert.
        Sind $ h : \m{M} \rightarrow \m{N} $, $ g : \m{N} \rightarrow \m{K} $ Isomorphismen, dann sind auch folgende Funktionen ein Isomorphimus:
        \begin{itemize}
            \item $ h^{-1} $
            \item $ g \circ h : \m{M} \rightarrow \m{K} $
        \end{itemize}

        $ h : \m{M} \Rightarrow \m{M} $ heißt \underline{Automorphismus von $ \m{M} $}, wenn $ h $ ein Isomorphimus ist.
        Die Struktur $ Aut(\m{M}) = \big( \{ h : \m{M} \rightarrow \m{M} \mid h \text{ Automorphismus} \}, \circ \big) $ mit Komposition $ \circ $ ist eine Gruppe.
\end{dfn}

\begin{dfn}
    Seien $ \m{M} = \struc{M}{\m{M}}{} $, $ \m{N} = \struc{N}{\m{N}}{} $ $\sigma $-Strukturen.
    $ \m{N} $ heißt \underline{Teil-/Unter-/Substruktur} von $ \m{M} $, falls:
    \begin{enumerate}
        \item $ N \subseteq M $
        \item $ c^\m{N} = c^\m{M} $ für $ c \in \m{C} $
        \item $ f^\m{N}(\bar{a}) = f^\m{M}(\bar{a}) $ für $ f \in \m{F} $, $ \bar{a} \in N^{\sigma'(f)} $
        \item $ \bar{a} \in R^\m{N} \Leftrightarrow \bar{a} \in R^\m{M} $, d.~h. $ R^\m{N} = R^\m{M} \cap N^{\sigma'(R)} $ für $ \bar{a} \in N^{\sigma'(R)} $
    \end{enumerate}

        Wenn $ \m{N} $ eine Teilstruktur von $ \m{M} $ ist, schreibt man auch $ \m{N} \subseteq \m{M} $.
        $ \m{M} $ wird dann auch \underline{Ober- oder Erweiterungsstruktur von $ \m{N} $} genannt.
\end{dfn}

\begin{dfn}
    Seien $ \sigma_0 = \sign{0} $, $ \sigma_1 = \sign{1} $ zwei Signaturen
    mit $ \sigma_0 \subseteq \sigma_1 $, d.~h. $ \m{C}_0 \subseteq \m{C}_1 $, $ \m{F}_0 \subseteq \m{F}_1 $, $ \m{R}_0 \subseteq \m{R}_1 $ und $ \sigma'_0 = \sigma'_1 \vert_{\m{F}_0 \cup \m{R}_0} $ .

    Sei $ \m{M} = \struc{M}{\m{M}}{1} $ eine $ \sigma_1 $-Struktur.
    Dann heißt $ \m{M} \vert_{\sigma_0} = \struc{M}{\m{M}}{0} $ das \underline{Redukt} von $ \m{M} $ auf $ \sigma_0 $ od. auch $ \sigma_0 $-Redukt.
    Umgekehrt heißt $ \m{M} $ \underline{Expansion} von $ \m{N} := \m{M} \vert_{\sigma_0} $ auf $ \sigma_1 $, d.~h. $ \m{M} $ entsteht aus $ \m{N} $ durch Hinzunahme geeigneter (bzgl. $ \sigma_1 $) Konstanten/Funktionen/Relationen.

    Eine Signatur $ \sigma = \sign{} $ heißt \underline{konstantenlos}, falls $ \m{C} = \emptyset $ und \underline{(rein) relational}, falls $ \m{C} = \m{F} = \emptyset $.
\end{dfn}

\begin{bsp}
    Betrachten wir die Gruppe $ \m{G} = (G, 1^\m{G}, \circ^\m{G}) $.
    $ \m{G} $ ist ein Redukt von $ (G, 1^\m{G}, \circ^\m{G}, \circ^{-1\m{G}}) $.
    Von $ \m{R} = (R, 0, 1, +, \cdot) $ ist $ (R, 0, +) $ ein Redukt.
\end{bsp}

\section{Sprachen und Formeln}

Wunsch: Formeln der Form $ \forall x. (x \geq 0 \rightarrow \exists y. x = y \cdot y) $.

Gegeben eine Signatur $ \sigma = \sign{} $.
\underline{Grundsymbole} der Sprache $ L = L(\sigma) $ sind:
\begin{enumerate}
    \item abzählbar viele Variablen $ v_n $, $ n \in \mathbb{N} $
    \item die Symbole von $ \sigma $ aus $ \m{C} \cup \m{F} \cup \m{R} $ (nichtlogische Symbole)
    \item Logisch Zeichen: $ \neg, \land, = $
    \item Quantor $ \exists $
    \item Klammersymbole $ ( $ und $ ) $
\end{enumerate}

Die Menge der \underline{Terme} von $ \sigma $ oder $ L(\sigma) $ ist die kleinste Menge für die gilt:
\begin{enumerate}
    \item Alle Variablen $ v_n $, $ n \in \mathbb{N} $ sind ein Term.
    \item Alle Konstantensymbole $ c \in \m{C} $ sind ein Term.
    \item Wenn $ t_1, ..., t_n $ Terme sind, $ f \in \m{F} $ mit $ \sigma'(f) = n $, dann ist auch $ f(t_1, ..., t_n) $ ein Term.
\end{enumerate}

Die Menge der \underline{Atomformeln} bzgl. $ \sigma $ ist die kleinste Menge für die gilt:
\begin{enumerate}
    \item Wenn $ t_1, t_2 $ Terme sind, dann ist $ t_1 = t_2 $ eine Atomformel,
    \item Wenn $ t_1, ..., t_n $ Terme sind, $ R \in \m{R} $ mit $ \sigma'(R) = n $, dann ist $ R(t_1, ..., t_n) $ eine Atomformel.
\end{enumerate}

Die Menge der \underline{Formeln} bzgl. $\sigma $ ist die kleineste Menge für die gilt:
\begin{enumerate}
    \item Jede Atomformel ist eine Formel.
    \item Wenn $ \varphi, \psi $ Formeln sind, $ x = v_n $ $ n \in \mathbb{N} $ Variable, dann sind auch $ \neg \varphi$, $ (\varphi \land \psi) $, $ (\exists x) \varphi $ Formeln.
\end{enumerate}

$ L(\sigma) := $ Menge der Formeln bzgl. $ \sigma $.
Abkürzungen für zwei Formeln $ \varphi , \psi \in L(\sigma) $ sind wie üblich definiert:
\begin{itemize}
    \item $ \varphi \lor \psi := \neg (\neg \varphi \land \neg \psi) $
    \item $ \varphi \rightarrow \psi := \neg \varphi \lor \psi $
    \item $ \varphi \leftrightarrow \psi := (\varphi \rightarrow \psi) \land (\psi \rightarrow \varphi) $
    \item $ (\forall x) \varphi := \neg (\exists x) \neg \varphi $
\end{itemize}

Die Grundsymbole sind von $ L(\sigma) $ sind absichtlich so spärlich definiert, da Beweise oftmals über den Aufbau von Formeln geführt werden und in diesem Fall minimal-viele Zeichen in Beweisen berücksichtigt werden müssen.

Seien $ x $ eine Variable $ \varphi $ und eine Formel. Der \underline{(Wirkungs-)Bereich} eines Quantors $ (\exists x) \varphi $ ist definiert als $ \varphi $.
Ein Vorkommen von $ x $ im Bereich von $ (\exists x) $ heißt \underline{gebunden}.
Ist $ x $ nicht im Bereich eines Quantors, so heißt dieses Vorkommen \underline{frei}.
$ x $ ist eine \underline{freie Variable}, wenn $ x $ an mindestens einer Stelle frei vorkommt; ansonsten heißt $ x $ \underline{gebunden}.

Eine Formel $ \varphi $ heißt \underline{Aussage} oder \underline{Satz}, falls $ \varphi $ keine freien Variablen enthält.

\end{document}
""".strip()

# Taken from: https://scholar.harvard.edu/files/grusdt/files/diploma-fgrusdt.pdf
# Fabian is one of the most fascinating and intelligent individuals I've ever met.
science_text = r"""
Der Austausch von Elektronen mit Photonen bedeutet, dass für DZPs das Pauli-Prinzip nicht gilt
und damit kein integraler Quanten-Hall-Effekt existiert. Zusätzliche repulsive Wechselwirkungen
können jedoch auch hier zu einer Mutation führen, so dass die wechselwirkenden Bosonen sich wie
schwach-wechselwirkende Fermionen in einem reduzierten Magnetfeld verhalten. Der fraktionale
Quanten-Hall-Effekt existiert also für Bosonen. Die starken repulsiven Wechselwirkungen können
durch langreichweitige Van-der-Waals Potentiale zwischen Rydberg-Zuständen realisiert werden
[63]. Dazu werden DZPs mit einem langlebigen Rydberg-Zustand betrachtet [59]. Es stellt sich
also die Frage nach der Form der effektiven Polariton-Polariton Wechselwirkung. Damit verbunden
muss geklärt werden, in welchen Parameterbereichen FQHE-Grundzustände vorliegen und ob ihre
Anregungslücken im messbaren Bereich liegen. In dieser Arbeit werden vor allem die prominenten
Laughlin-Zustände untersucht.

Verschiedene Ansätze versuchen den bosonischen FQHE in ultra-kalten Quantengasen nachzuweisen, wozu rotierende Bose-Einstein-Kondensate verwendet werden [68]. Sollte dies gelingen, so
ergeben sich vielfältige Manipulationsmöglichkeiten der elementaren Anregungen. Unter anderem
ist eine direkte Vermessung der anyonischen Statistik möglich [60]. Im Folgenden soll untersucht
werden, ob in einer quantenoptischen Realisierung ein ebenso großes Maß an Kontrolle vorhanden
ist. Es stellt sich insbesondere die Frage, wie die gewünschten Zustände präpariert werden können.
Dazu kommt die Verwendung von Dunkelzuständen nichtlinearer Polaritonen- Vernichtungsoperatoren in offenen Quantensystemen in Frage.
""".strip()

# The tokenizer was trained on data from earlier shards, so it has seen this data
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# Log to report
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
