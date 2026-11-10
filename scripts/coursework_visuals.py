# ruff: noqa: E501, RUF001
"""Shared chart, glossary and infrastructure components for the local viewers."""

from html import escape

TERMS = {
    "Токен": "Единица текста, которую распознаёт модель: слово, часть слова или знак.",
    "Вектор": "Упорядоченный набор чисел. Номер координаты не означает номер слова.",
    "Слот": "Ячейка общего рабочего пространства. Хранит вектор, источник, возраст и время жизни.",
    "Модуль": "Вычислительный блок со своим состоянием. В цикле работают все восемь модулей; "
    "право записи получают только две заявки.",
    "Заявка": "Вектор и оценки, которые модуль предлагает записать. "
    "Собственная оценка заявки не равна итоговому баллу арбитра.",
    "Вес чтения": "Доля вклада слота во взвешенную сумму, прочитанную модулем. "
    "Это не оценка важности слова и не доказательство причинного влияния.",
    "Норма": "Длина вектора: квадратный корень из суммы квадратов его координат. "
    "Большая норма сама по себе не означает более полезную запись.",
    "Время жизни": "Счётчик до начала затухания содержимого слота. "
    "Нулевое значение не обязательно означает немедленное удаление.",
    "Векторный мост": "Обученный блок, который преобразует содержимое каждого слота "
    "в вектор ширины входа Qwen. Веса Qwen и когнитивные модули не обучались.",
    "Маска внимания": "Признак, участвует ли позиция во внимании. "
    "Это не то же самое, что занятость соответствующего слота.",
}

NODES = (
    ("input", "Ввод и шаблон", "Запрос → последовательность токенов"),
    ("qwen", "Qwen: первый проход", "Скрытые состояния текста"),
    ("route", "Подготовка контекста", "Выбор слоёв, вектор цели, распределение входов"),
    ("decay", "Затухание", "Возраст, время жизни и очистка занятых слотов"),
    ("read", "Чтение", "Занятые слоты → взвешенное чтение"),
    ("modules", "8 модулей", "Обновляют независимые состояния"),
    ("proposal", "Заявки", "Каждый модуль предлагает вектор"),
    ("select", "Арбитр и распределитель", "Две заявки → два разных слота"),
    ("write", "Запись", "Новые векторы и метаданные слотов"),
    ("bridge-check", "Промежуточный проход", "Внутренний проход оркестратора до учебного моста"),
    ("bridge", "Векторный мост", TERMS["Векторный мост"]),
    ("decode", "Qwen: ответ", "Генератор получает текстовые позиции и векторы"),
)
NODE_IDS = frozenset(node for node, _, _ in NODES)


def component_for_stage(name: str) -> str | None:
    for prefix, component in (
        ("frozen first pass", "qwen"),
        ("backbone layer sampling", "route"),
        ("request goal encoding", "route"),
        ("module input routing", "route"),
        ("workspace decay before cycle ", "decay"),
        ("workspace read for cycle ", "read"),
        ("cognitive modules for cycle ", "modules"),
        ("proposal head for cycle ", "proposal"),
        ("workspace arbitration for cycle ", "select"),
        ("slot allocation for cycle ", "select"),
        ("workspace write for cycle ", "write"),
        ("workspace latent bridge", "bridge-check"),
        ("second-pass generation prefix", "bridge-check"),
    ):
        if name.startswith(prefix):
            return component
    return None


ATLAS_HTML = (
    """
<section class="visual-panel" aria-label="Карта обработки">
<div class="section-label">01 · Путь запроса</div>
<h2>Карта вычислений</h2>
<p id="atlas-status" role="status">Схема показывает последовательность обработки запроса.</p>
<div class="atlas-track" aria-label="Последовательность узлов">
"""
    + "".join(
        f'<div class="atlas-node" data-component="{key}" tabindex="0" '
        f'title="{escape(hint)}" aria-label="{escape(label + ": " + hint)}">'
        f'<strong><span class="atlas-number">{index:02d}</span> {escape(label)}</strong>'
        f"<small>{escape(hint)}</small></div>"
        for index, (key, label, hint) in enumerate(NODES, 1)
    )
    + """
</div><p class="atlas-loop">↺ Затухание → чтение → модули → отбор → запись. Три цикла; в первом затухания нет.</p>
<details class="glossary"><summary>Словарь терминов</summary><dl>
"""
    + "".join(
        f"<dt tabindex='0' title='{escape(hint)}'>{escape(term)}</dt><dd>{escape(hint)}</dd>"
        for term, hint in TERMS.items()
    )
    + """</dl></details></section>
"""
)

VISUAL_STYLE = """
:root{--iris-ink:#16213b;--iris-paper:#f3f6fb;--iris-card:#fff;--iris-border:#d2dbea;
 --iris-accent:#245db3;--iris-contrast:#00806d;--iris-muted:#435978}
@media(prefers-color-scheme:dark){:root{--iris-ink:#edf3ff;--iris-paper:#101725;
 --iris-card:#182339;--iris-border:#3a4a65;--iris-accent:#8fb9ff;
 --iris-contrast:#71e0c1;--iris-muted:#b7c8e5}}
body{background:var(--iris-paper)!important;color:var(--iris-ink)!important}
h1{letter-spacing:-.035em}h2{letter-spacing:-.022em}
.visual-panel{margin:16px 0;padding:16px;border:1px solid var(--iris-border);
 border-radius:14px;background:var(--iris-card)}
.section-label{font-size:.78rem;color:var(--iris-muted);letter-spacing:.08em}
.visual-panel h2{margin-top:4px}
.atlas-track{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.atlas-node{border:1px solid var(--iris-border);border-radius:10px;padding:9px;min-width:0}
.atlas-node strong,.atlas-node small{display:block}.atlas-node small{color:var(--iris-muted);margin-top:4px}
.atlas-number{font-size:.72em;color:var(--iris-accent);font-variant-numeric:tabular-nums}
.atlas-node[data-visited=true]{border-color:var(--iris-contrast)}
.atlas-node[data-active=true]{outline:2px solid var(--iris-accent);outline-offset:-2px;
 background:color-mix(in srgb,var(--iris-accent) 14%,var(--iris-card))}
.atlas-node:focus-visible,summary:focus-visible{outline:2px solid var(--iris-accent)}
.atlas-loop{color:var(--iris-muted);font-size:.86rem}
.glossary dl{display:grid;grid-template-columns:minmax(105px,1fr) 4fr;gap:7px}
.glossary dd{margin:0}.glossary dt{font-weight:650}
abbr[title]{cursor:help;text-decoration:underline dotted;text-underline-offset:3px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:10px 0}
.metric-chart{margin:0;padding:10px;border:1px solid var(--iris-border);border-radius:9px;min-width:0}
.metric-chart svg{width:100%;height:auto;display:block}
.metric-chart text{fill:var(--iris-ink);font:12px system-ui}
.metric-chart .axis{stroke:var(--iris-border);stroke-width:1}
.metric-chart .proposal{stroke:var(--iris-accent);fill:none;stroke-width:1.8}
.metric-chart .stored{stroke:var(--iris-contrast);fill:none;stroke-width:1.8;stroke-dasharray:4 2}
.metric-chart .bar{fill:var(--iris-accent)}.metric-chart .chosen-bar{fill:var(--iris-contrast)}
.raw-data{margin:12px 0}.raw-data summary{cursor:pointer;font-weight:650}
.vector-legend{font-size:.88rem;color:var(--iris-muted)}
"""

ATLAS_JS = """
function drawAtlas(active, visited, status){
 document.querySelectorAll("[data-component]").forEach(node=>{
  node.dataset.active=String(node.dataset.component===active);
  node.dataset.visited=String(visited.includes(node.dataset.component));
 });
 document.getElementById("atlas-status").textContent=status;
}
"""

CHART_JS = """
const gw=id=>document.getElementById(id);
function graphNode(parent,tag,attributes={},label=""){
 const node=document.createElementNS("http://www.w3.org/2000/svg",tag);
 for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
 if(label)node.textContent=label;
 parent.appendChild(node);return node;
}
function chartCard(title,hint){
 const figure=document.createElement("figure"), caption=document.createElement("figcaption");
 figure.className="metric-chart";caption.textContent=title;caption.title=hint;
 caption.tabIndex=0;figure.append(caption);
 return [figure,graphNode(figure,"svg",{viewBox:"0 0 340 126",role:"img","aria-label":title})];
}
function vectorChart(pair){
 const [figure,svg]=chartCard("Модуль "+pair.module+" → слот "+pair.slot,
  "256 координат в исходном порядке. Обе линии используют одну вертикальную шкалу.");
 const max=Math.max(1e-9,...pair.proposal.map(Math.abs),...pair.stored.map(Math.abs));
 const y=v=>52-42*v/max, x=(i,n)=>22+302*i/(n-1);
 graphNode(svg,"line",{x1:22,y1:52,x2:324,y2:52,class:"axis"});
 for(const [name,values] of [["proposal",pair.proposal],["stored",pair.stored]]){
  graphNode(svg,"polyline",{class:name,points:values.map((v,i)=>
   x(i,values.length).toFixed(2)+","+y(v).toFixed(2)).join(" ")});
 }
 graphNode(svg,"text",{x:22,y:116},"0");
 graphNode(svg,"text",{x:324,y:116,"text-anchor":"end"},"255 · номер координаты");
 graphNode(svg,"text",{x:22,y:105},"±"+max.toPrecision(3)+" · общая шкала");
 return figure;
}
function barChart(title,hint,values,selected=[],signed=false){
 const [figure,svg]=chartCard(title,hint);
 const max=Math.max(1e-9,...values.map(Math.abs)), zero=signed?53:91;
 const scale=(signed?38:76)/max, step=300/values.length;
 graphNode(svg,"line",{x1:20,y1:zero,x2:322,y2:zero,class:"axis"});
 values.forEach((v,i)=>{
  const height=Math.abs(v)*scale;
  const bar=graphNode(svg,"rect",{x:22+i*step,y:v<0?zero:zero-height,width:step*.65,
   height:Math.max(.5,height),rx:2,class:selected.includes(i)?"chosen-bar":"bar"});
  graphNode(bar,"title",{},"№ "+i+": "+v.toPrecision(4));
  graphNode(svg,"text",{x:22+(i+.3)*step,y:108,"text-anchor":"middle"},i);
 });
 return figure;
}
function drawLatentChart(id,latent){
 const target=gw(id); if(!target)return;
 if(!latent){target.replaceChildren();return;}
 target.replaceChildren(barChart("Векторы на входе Qwen · длины",
  "По одному вектору на слот; это результат работы моста после последнего цикла.",
  latent.norms,latent.slot_indices.filter((_,i)=>latent.slot_occupancy[i])));
}
"""
