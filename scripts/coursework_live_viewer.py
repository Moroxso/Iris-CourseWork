"""Read-only, per-session browser view of the interactive coursework run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import hypot, isclose, isfinite
from secrets import token_urlsafe
from threading import Lock, Thread
from typing import Any


def validate_vectors(row: dict[str, Any], vectors: Any) -> None:
    """Check that the vectors correspond to this cycle's actual write order."""
    if not isinstance(vectors, list) or len(vectors) != len(row["selected_slots"]):
        raise ValueError("Missing winning vectors")
    for pair, module, slot in zip(
        vectors, row["write_modules"], row["selected_slots"], strict=True
    ):
        if not isinstance(pair, dict) or pair.get("module") != module or pair.get("slot") != slot:
            raise ValueError("Winning vector write pair mismatch")
        for name in ("proposal", "stored"):
            values = pair.get(name)
            if (
                not isinstance(values, list)
                or len(values) != 256
                or any(type(v) not in (int, float) or not isfinite(v) for v in values)
            ):
                raise ValueError("Invalid workspace vector coordinates")
        if not isclose(hypot(*pair["stored"]), row["slot_norms"][slot], rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("Stored vector norm does not match the recorded slot")


class LiveState:
    """Only the current request and bounded diagnostics; no paths or history."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._current: dict[str, Any] = {}
        self._wire = b""
        self._revision = 0
        self.publish("ready", {})

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            next_state = dict(self._current)
            if event in {"ready", "request"}:
                next_state = dict(
                    request=payload.get("request"),
                    prompt="",
                    answer="",
                    detail="",
                    cycles=[],
                    cycle_count=None,
                    latents=None,
                )
            elif event == "first-pass":
                next_state["prompt"] = payload["prompt"]
            elif event == "cycle":
                cycles = next_state["cycles"]
                row = payload["row"]
                count = payload["cycle_count"]
                if (
                    not 1 <= row["cycle"] == len(cycles) + 1 <= count
                    or count > 32
                    or next_state["cycle_count"] not in (None, count)
                ):
                    raise ValueError("Live view requires ordered workspace cycles")
                modules, slots = row["write_modules"], row["selected_slots"]
                if (
                    len(modules) != len(slots)
                    or sorted(modules) != row["selected_modules"]
                    or len(set(modules)) != len(modules)
                    or len(set(slots)) != len(slots)
                    or any(
                        not 0 <= slot < len(row["slot_sources"])
                        or row["slot_sources"][slot] != module
                        for module, slot in zip(modules, slots, strict=True)
                    )
                ):
                    raise ValueError("Live write pairs must match the observed workspace")
                if "winning_vectors" in row:
                    validate_vectors(row, row["winning_vectors"])
                next_state.update(cycles=[*cycles, row], cycle_count=count)
            elif event == "generation":
                latents = payload["latents"]
                if (
                    latents["source_cycle"] != next_state["cycle_count"]
                    or len(next_state["cycles"]) != latents["source_cycle"]
                ):
                    raise ValueError("Live latents require the complete workspace trace")
                next_state["latents"] = latents
            elif event == "completed":
                if next_state["latents"] is None:
                    raise ValueError("Live result has no observed generator input")
                next_state.update(answer=payload["text"], detail=payload["stop_reason"])
            elif event in {"rejected", "failed"}:
                next_state["detail"] = payload["message"]
            else:
                raise ValueError("Unknown live workspace event")
            next_state.update(stage=event, revision=self._revision + 1)
            wire = json.dumps(next_state, ensure_ascii=False, allow_nan=False).encode("utf-8")
            # Isolate the snapshot from dictionaries owned by the inference code.
            self._current = json.loads(wire)
            self._wire = wire
            self._revision += 1

    def snapshot(self) -> bytes:
        with self._lock:
            return self._wire


@contextmanager
def live_view(state: LiveState) -> Iterator[str]:
    """Serve just two GET routes on IPv4 loopback for the console's lifetime."""
    secret = "/" + token_urlsafe(24)
    home = secret + "/"
    page = PAGE.encode("utf-8")
    host = ""

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(1)

        def log_message(self, _format: str, *_args: Any) -> None:
            # Neither the session URL nor the request is written to HTTP logs.
            return

        def do_GET(self) -> None:
            hosts = self.headers.get_all("Host", [])
            if hosts != [host] or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.send_error(403)
                return
            if self.path == home:
                content_type, body = "text/html; charset=utf-8", page
            elif self.path == home + "state":
                content_type, body = "application/json; charset=utf-8", state.snapshot()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'",
            )
            # A closed/refreshed viewer is not a failed model request.
            with suppress(BrokenPipeError, ConnectionResetError, TimeoutError):
                self.end_headers()
                self.wfile.write(body)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        host = f"127.0.0.1:{server.server_port}"
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        try:
            yield "http://" + host + home
        finally:
            server.shutdown()
            thread.join()


GRAPH_STYLE = """
.gw-graph{width:100%;min-width:660px;height:auto;display:block}
.gw-graph text{fill:CanvasText;font:15px system-ui}
.gw-graph rect{fill:Canvas;stroke:GrayText;stroke-width:1.2}
.gw-graph .chosen{fill:color-mix(in srgb,Highlight 22%,Canvas);stroke:Highlight;stroke-width:2.5}
.gw-graph .write{stroke:Highlight;stroke-width:3;fill:none}
.gw-graph .read{stroke:CanvasText;stroke-width:1.5;stroke-dasharray:4 5;fill:none}
#gw-reads,#gw-vectors{margin:6px 0}
#gw-vectors details{padding:6px 10px;border-left:3px solid Highlight;margin:6px 0}
#gw-vectors pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85em}
"""

GRAPH_HTML = """
<section id="gw-diagram" hidden>
<h2>Обмен между модулями и слотами</h2>
<p>Сплошные стрелки: запись победителей. Пунктир: чтение выбранным модулем до записи.</p>
<label for="gw-reader">Показать чтение:</label>
<select id="gw-reader"></select>
<div class="scroll"><svg class="gw-graph" id="gw-graph" viewBox="0 0 960 174"
 role="img" aria-label="Схема связей модулей и слотов"></svg></div>
<p id="gw-reads"></p>
<h2>Векторы победивших заявок</h2>
<p>Побеждают заявки модулей, не слова. Каждый вектор содержит 256 числовых координат.
 Показаны предложенный вектор и содержимое слота после записи: они могут различаться.</p>
<div id="gw-vectors"></div>
</section>
"""

GRAPH_JS = """
const gw=id=>document.getElementById(id);
let gwCycle=null, gwRequest=null;
function graphNode(parent,tag,attributes={},label=""){
 const node=document.createElementNS("http://www.w3.org/2000/svg",tag);
 for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
 if(label)node.textContent=label;
 parent.appendChild(node);return node;
}
function drawGraphLinks(){
 const c=gwCycle, svg=gw("gw-graph"), reader=Number(gw("gw-reader").value);
 svg.replaceChildren();
 const defs=graphNode(svg,"defs");
 for(const name of ["read","write"]){
  const marker=graphNode(defs,"marker",{
   id:"gw-"+name,viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:5,markerHeight:5,orient:"auto"});
  graphNode(marker,"path",{d:"M 0 0 L 10 5 L 0 10 z",fill:name==="read"?"CanvasText":"Highlight"});
 }
 const x=(i,n)=>960*(i+.5)/n, modules=c.module_bids.length, slots=c.slot_sources.length;
 const reads=c.read_attention[reader].map((weight,slot)=>({weight,slot})).filter(
  ({weight,slot})=>weight>0&&c.read_slot_occupancy[slot]);
 for(const {slot,weight} of reads){
  const edge=graphNode(svg,"line",{
   x1:x(slot,slots)-9,y1:128,x2:x(reader,modules)-9,y2:46,
   class:"read","marker-end":"url(#gw-read)"});
  graphNode(edge,"title",{},"Чтение слота "+slot+", вес "+weight.toPrecision(4));
 }
 c.selected_slots.forEach((slot,i)=>{
  const edge=graphNode(svg,"line",{
   x1:x(c.write_modules[i],modules)+9,y1:46,x2:x(slot,slots)+9,y2:128,
   class:"write","marker-end":"url(#gw-write)"});
  graphNode(edge,"title",{},"Модуль "+c.write_modules[i]+" → слот "+slot);
 });
 for(const [name,count,y,selected] of [
  ["Модуль",modules,10,c.selected_modules],["Слот",slots,128,c.selected_slots]]){
  for(let i=0;i<count;i++){
   const mid=x(i,count);
   graphNode(svg,"rect",{x:mid-51,y,width:102,height:36,rx:6,
    class:selected.includes(i)?"chosen":""});
   graphNode(svg,"text",{x:mid,y:y+23,"text-anchor":"middle"},name+" "+i);
  }
 }
 gw("gw-reads").textContent=reads.length?
  "Модуль "+reader+" читал: "+reads.map(({slot,weight})=>{
   const source=c.read_slot_sources[slot];
   return "слот "+slot+" (источник: модуль "+source+", вес "+weight.toPrecision(3)+")";
  }).join(" · "):"Модуль "+reader+" не читал ни одного занятого слота в этом цикле.";
}
function drawWorkspaceGraph(c,request){
 const enriched=Boolean(c?.winning_vectors);
 gw("gw-diagram").hidden=!enriched;
 if(!enriched){gwCycle=null;gwRequest=null;return;}
 if(gwRequest===request&&gwCycle?.cycle===c.cycle)return;
 const select=gw("gw-reader");
 if(gwRequest!==request||!gwCycle){
  select.replaceChildren(...c.module_bids.map((_,i)=>new Option("Модуль "+i,String(i))));
  select.value=String(c.write_modules[0]??0);
 }
 gwRequest=request;gwCycle=c;
 drawGraphLinks();
 gw("gw-vectors").replaceChildren(...c.winning_vectors.map(pair=>{
  const detail=document.createElement("details"), summary=document.createElement("summary");
  summary.textContent="Модуль "+pair.module+" → слот "+pair.slot+" · "+pair.proposal.length+
   " координат в каждом векторе";
  const preview=document.createElement("p");
  const first=values=>values.slice(0,4).map(v=>v.toPrecision(5)).join(", ");
  preview.textContent="Первые четыре координаты: заявка ["+first(pair.proposal)+
   "]; после записи ["+first(pair.stored)+"].";
  detail.append(summary,preview);
  for(const [title,values] of [["Предложенный вектор",pair.proposal],["После записи",pair.stored]]){
   const heading=document.createElement("strong"), coordinates=document.createElement("pre");
   heading.textContent=title;coordinates.textContent=values.join(", ");
   detail.append(heading,coordinates);
  }
  return detail;
 }));
}
gw("gw-reader").addEventListener("change",()=>{if(gwCycle)drawGraphLinks();});
"""

PAGE = (
    """<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris · живой Global Workspace</title>
<style>
:root{color-scheme:light dark;font:16px system-ui}
body{max-width:960px;margin:auto;padding:24px;background:Canvas;color:CanvasText}
h1{font-size:1.55rem}h2{font-size:1.1rem;margin-top:24px}
p{white-space:pre-wrap;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 6px;text-align:left;border-bottom:1px solid GrayText}
.scroll{overflow-x:auto}input[type=range]{width:100%}
.notice{border-left:4px solid GrayText;padding:8px 12px}
caption{text-align:left;margin:8px 0}
.top{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px 16px}
.cycle-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px}
.cycle-card{background:Canvas;color:CanvasText;border:1px solid GrayText;border-radius:6px;
 text-align:left;padding:8px 10px;font:inherit}
.cycle-card strong,.cycle-card span{display:block}
.cycle-card span{font-size:.88em}
.cycle-card[aria-pressed=true]{border-color:Highlight;box-shadow:inset 0 0 0 1px Highlight}
.cycle-card:disabled{opacity:.66}
.written{background:color-mix(in srgb,Highlight 19%,Canvas)}
.written td:first-child{box-shadow:inset 3px 0 Highlight;font-weight:700;padding-left:10px}
.compact .details-only{display:none}
.compact{max-width:1240px;padding:12px 20px}
.compact h1,.compact h2,.compact p{margin:6px 0}
.compact .notice{padding-block:4px}
.compact table{font-size:.87rem}
.compact th,.compact td{padding-block:4px}
.compact #cycle-controls{display:none}
.compact .request-pair,.compact .tables{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px}
.compact .tables>section,.compact .request-pair>section{min-width:0}
#connection:empty{display:none}
@media(max-width:820px){.compact .tables,.compact .request-pair{grid-template-columns:1fr}}
"""
    + GRAPH_STYLE
    + """
</style>
<div class="top"><h1>Iris · живой Global Workspace</h1>
<label><input id="compact" type="checkbox" checked> Компактный вид</label></div>
<p>Введите промпт в консоли. Здесь отображается текущий запрос, без истории чата.</p>
<p class="notice" id="stage" role="status">Подключаюсь к сеансу…</p>
<p id="connection" role="status"></p>
<div class="request-pair">
<section><h2>Запрос</h2><p id="prompt">—</p></section>
<section><h2>Ответ Iris</h2><p id="answer">—</p></section>
</div>
<h2>Циклы рабочего пространства</h2>
<div class="cycle-list" id="cycle-list" aria-label="Выбор цикла"></div>
<label><input id="follow" type="checkbox" checked> Следовать за последним циклом</label>
<p id="cycle-controls"><label for="cycle">Выбрать цикл</label>
<input id="cycle" type="range" min="0" max="0" value="0" step="1" disabled></p>
<p>Показан цикл <output id="cycle-number">—</output></p>
<p id="selection">Данные появятся после записи первого цикла.</p>
"""
    + GRAPH_HTML
    + """
<div class="scroll details-only"><table><caption>Модули: заявка на запись</caption>
<thead><tr><th>Модуль</th><th>Bid</th><th>Confidence</th><th>Выбран</th></tr></thead>
<tbody id="modules"></tbody></table></div>
<div class="tables"><section>
<h2>Слоты после записи</h2>
<div class="scroll"><table><caption>Подсвечены записи выбранного цикла</caption>
<thead><tr><th>Слот</th><th>Занят</th><th>Источник</th><th>Норма</th><th>Возраст</th><th>TTL</th>
<th>Запись</th></tr></thead>
<tbody id="slots"></tbody></table></div>
</section><section>
<h2>Latent embeddings перед генерацией</h2>
<p id="latent-summary">Появятся при передаче входа генератору, после последнего цикла.</p>
<div class="scroll"><table><caption>Фактический вход генератора, независимо от ползунка</caption>
<thead><tr><th>Вектор</th><th>Слот</th><th>Занят</th><th>Источник</th>
<th>Позиция</th><th>Mask</th><th>L2 норма</th></tr></thead>
<tbody id="latents"></tbody></table></div>
</section></div>
<script>
"use strict";
"""
    + GRAPH_JS
    + """
const byId=id=>document.getElementById(id);
const set=(id,value)=>{byId(id).textContent=value;};
const slider=byId("cycle"), follow=byId("follow"), compact=byId("compact");
const num=v=>v.toFixed(4), yes=v=>v?"да":"нет";
const stages={
 ready:"Ожидаю промпт в консоли",request:"Проверяю промпт",
 "first-pass":"Первый проход: вычисляю контекст",cycle:"Рабочее пространство: циклы",
 generation:"Векторы переданы генератору; вычисляю ответ",
 completed:"Готово",rejected:"Запрос отклонён",failed:"Ошибка запроса"
};
let state=null;
function rows(id,data,written=[]){
 const body=byId(id);
 body.replaceChildren(...data.map((values,i)=>{
  const tr=document.createElement("tr");
  if(written.includes(i))tr.className="written";
  for(const value of values){
   const td=document.createElement("td");td.textContent=String(value);tr.appendChild(td);
  }
  return tr;
 }));
}
function writePairs(c){
 return c.selected_slots.map((slot,i)=>"Модуль "+c.write_modules[i]+" → слот "+slot);
}
function drawCards(){
 const group=byId("cycle-list");
 const signature=JSON.stringify(state.cycles.map(c=>writePairs(c)))+":"+state.cycle_count;
 if(group.dataset.cycles!==signature){
  group.dataset.cycles=signature;
  group.replaceChildren(...Array.from({length:state.cycle_count??0},(_,i)=>{
   const button=document.createElement("button");
   button.type="button";button.className="cycle-card";
   const name=document.createElement("strong"), pairs=document.createElement("span");
   name.textContent="Цикл "+(i+1);
   const c=state.cycles[i];
   pairs.textContent=c?writePairs(c).join(" · ")||"Нет записей":"Ожидание";
   button.disabled=!c;button.append(name,pairs);
   button.addEventListener("click",()=>{
    slider.value=String(i);follow.checked=false;drawCycle();
   });
   return button;
  }));
 }
 Array.from(group.children).forEach((button,i)=>button.setAttribute(
  "aria-pressed",String(Boolean(state.cycles[i])&&Number(slider.value)===i)));
}
function drawCycle(){
 if(state)drawCards();
 const c=state?.cycles[Number(slider.value)];
 drawWorkspaceGraph(c,state?.request);
 if(!c){
  set("cycle-number","—");set("selection","Нет записанных циклов для этого запроса.");
  rows("modules",[]);rows("slots",[]);return;
 }
 set("cycle-number",c.cycle+" / "+state.cycle_count);
 set("selection","Нумерация модулей и слотов начинается от нуля.");
 rows("modules",c.module_bids.map((bid,i)=>
  [i,num(bid),num(c.module_confidence[i]),yes(c.selected_modules.includes(i))]));
 rows("slots",c.slot_norms.map((norm,i)=>
  [i,yes(c.slot_occupancy[i]),c.slot_occupancy[i]?c.slot_sources[i]:"—",
   num(norm),c.slot_ages[i],num(c.slot_ttl[i]),c.selected_slots.includes(i)?"Да":"—"]),
   c.selected_slots);
}
function draw(next){
 if(state?.revision===next.revision)return;
 const fresh=state?.request!==next.request||next.stage==="request";
 state=next;
 const count=next.cycles.length;
 let status=stages[next.stage];
 if(next.request!==null)status="Запрос "+next.request+" · "+status;
 if(next.stage==="cycle")status+=" "+count+" / "+next.cycle_count;
 if(next.detail)status+=" · "+next.detail;
 set("stage",status);
 set("prompt",next.prompt||"—");
 set("answer",next.stage==="completed"?next.answer||"(пустой ответ)":"—");
 slider.max=String(Math.max(0,count-1));slider.disabled=count===0;
 if(fresh)follow.checked=true;
 if(follow.checked)slider.value=slider.max;
 drawCycle();
 const l=next.latents;
 set("latent-summary",l?
  l.num_tokens+" \u00d7 "+l.embedding_dim+" · цикл "+l.source_cycle+" · "+l.dtype:
  "Нет векторов для текущего запроса. Они появятся на входе генератора.");
 rows("latents",l?l.slot_indices.map((s,i)=>
  [i,s,yes(l.slot_occupancy[i]),l.slot_occupancy[i]?l.slot_sources[i]:"—",
   l.positions[i],String(l.attention_mask[i]),num(l.norms[i])]):[]);
}
slider.addEventListener("input",()=>{follow.checked=false;drawCycle();});
follow.addEventListener("change",()=>{if(follow.checked){slider.value=slider.max;drawCycle();}});
function layout(){document.body.classList.toggle("compact",compact.checked);}
compact.addEventListener("change",layout);
layout();
async function poll(){
 let delay=120;
 try{
  const response=await fetch("state",{cache:"no-store",signal:AbortSignal.timeout(1500)});
  if(!response.ok)throw new Error("Live HTTP status "+response.status);
  draw(await response.json());set("connection","");
 }catch{
  set("connection","Консоль недоступна. Последние показанные данные не обновляются.");
  delay=850;
 }
 setTimeout(poll,delay);
}
poll();
</script>
</html>
"""
)
