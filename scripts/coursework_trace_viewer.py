"""Build an offline viewer from actual coursework result and workspace trace."""

import argparse
import json
import re
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any


def validate_latent_metadata(
    metadata: dict[str, Any], trace: dict[str, Any], identity: Any
) -> None:
    count = trace["num_slots"]
    if (
        metadata.get("kind") != "iris-coursework-latent-token-metadata-v1"
        or metadata.get("observed_at") != "IrisGenerator.generate-input"
        or type(metadata.get("source_cycle")) is not int
        or type(metadata.get("num_tokens")) is not int
        or metadata.get("source_cycle") != trace["cycle_count"]
        or metadata.get("num_tokens") != count
        or metadata.get("dtype") != "torch.float32"
        or type(metadata.get("embedding_dim")) is not int
        or metadata["embedding_dim"] < 1
        or metadata.get("vocabulary_token_ids") is not False
    ):
        raise ValueError("Observed final-cycle latent metadata required")
    for field in ("positions", "slot_indices", "slot_sources", "slot_occupancy", "attention_mask"):
        vector = metadata.get(field)
        required = bool if field in {"slot_occupancy", "attention_mask"} else int
        if (
            not isinstance(vector, list)
            or len(vector) != count
            or any(type(v) is not required for v in vector)
        ):
            raise ValueError("Typed latent metadata vector required")
    start, stop, prompt, length = (
        metadata.get(k)
        for k in ("span_start", "span_end_exclusive", "prompt_length", "prefix_length")
    )
    if (
        type(start) is not int
        or type(stop) is not int
        or type(prompt) is not int
        or type(length) is not int
    ):
        raise ValueError("Integer latent span boundaries required")
    if (
        not 0 < start < prompt
        or stop != start + count
        or length != prompt + count
        or stop > length
        or metadata.get("positions") != list(range(start, stop))
        or metadata.get("slot_indices") != list(range(count))
        or metadata.get("attention_mask") != [True] * count
    ):
        raise ValueError("Latent span/position/mask mismatch")
    final = trace["cycles"][-1]
    if (
        metadata.get("slot_occupancy") != final["slot_occupancy"]
        or metadata.get("slot_sources") != final["slot_sources"]
    ):
        raise ValueError("Latent metadata does not match final workspace slots")
    norms = metadata.get("norms")
    if (
        not isinstance(norms, list)
        or len(norms) != count
        or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in norms)
        or not isinstance(metadata.get("embedding_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", metadata["embedding_sha256"]) is None
    ):
        raise ValueError("Finite latent norms and vector content hash required")
    raw = (
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    if (
        not isinstance(identity, dict)
        or identity.get("bytes") != len(raw)
        or identity.get("sha256") != sha256(raw).hexdigest()
    ):
        raise ValueError("Latent metadata content binding mismatch")


def render_report(result: dict[str, Any], trace: dict[str, Any]) -> str:
    if result.get("completed") is not True or result.get("kind") not in {
        "iris-coursework-two-pass-demo-v1",
        "iris-coursework-trained-bridge-demo-v1",
    }:
        raise ValueError("Completed coursework result required")
    if result["kind"] == "iris-coursework-trained-bridge-demo-v1":
        checkpoint = result.get("checkpoint")
        if (
            result.get("bridge_trained") is not True
            or result.get("cognitive_components_trained") is not False
            or result.get("sidecar_fully_trained") is not False
            or result.get("backbone_frozen") is not True
            or result.get("source_provenance_verified") is not True
            or type(result.get("optimizer_updates_performed")) is not int
            or result.get("optimizer_updates_performed") != 0
            or not isinstance(checkpoint, dict)
        ):
            raise ValueError("Explicit bridge-only training provenance required")
        steps = checkpoint.get("completed_steps")
        if type(steps) is not int or not 1 <= steps <= 12:
            raise ValueError("Bounded checkpoint step required")
        for field in ("manifest", "state"):
            descriptor = checkpoint.get(field)
            if (
                not isinstance(descriptor, dict)
                or not isinstance(descriptor.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is None
            ):
                raise ValueError("Checkpoint content hashes required")
        raw_trace = (
            json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        identity = result.get("workspace_trace")
        options = trace.get("options", {})
        if (
            not isinstance(identity, dict)
            or identity.get("sha256") != sha256(raw_trace).hexdigest()
            or identity.get("bytes") != len(raw_trace)
            or not isinstance(options, dict)
            or options.get("checkpoint_id") != checkpoint["manifest"]["sha256"]
            or options.get("seed") != result.get("seed")
        ):
            raise ValueError("Workspace trace content/checkpoint binding mismatch")
    elif (
        result.get("bridge_trained", False) is not False
        or result.get("sidecar_trained", False) is not False
    ):
        raise ValueError("Untrained result kind cannot claim trained components")
    cycles = trace.get("cycles")
    if trace.get("kind") != "iris-workspace-diagnostic-trace-v1" or not isinstance(cycles, list):
        raise ValueError("Actual workspace diagnostic trace required")
    if not cycles or len(cycles) != trace.get("cycle_count"):
        raise ValueError("Complete nonempty cycle list required")
    for cycle in cycles:
        for field in ("slot_sources", "slot_occupancy", "slot_norms", "slot_ages", "slot_ttl"):
            if len(cycle[field]) != trace["num_slots"]:
                raise ValueError("Slot vector length mismatch")
        for field in ("module_bids", "module_confidence"):
            if len(cycle[field]) != trace["num_modules"]:
                raise ValueError("Module vector length mismatch")
    if result.get("latent_tokens") is not None:
        if not isinstance(result["latent_tokens"], dict):
            raise ValueError("Latent metadata object required")
        validate_latent_metadata(result["latent_tokens"], trace, result.get("latent_trace"))
    data = json.dumps(dict(result=result, trace=trace), ensure_ascii=True, allow_nan=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.replace("DATA_PLACEHOLDER", data)


TEMPLATE = """<!doctype html>
<html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>Iris — фактический workspace trace</title>
<style>
:root{color-scheme:light dark;font:16px system-ui}
body{max-width:960px;margin:auto;padding:24px;background:Canvas;color:CanvasText}
h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:24px}
p{white-space:pre-wrap;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px 6px;text-align:left;border-bottom:1px solid GrayText}
.scroll{overflow-x:auto}input{width:100%}
.notice{border-left:4px solid GrayText;padding-left:12px}
caption{text-align:left;margin-bottom:8px}
</style>
<h1>Iris · фактический Global Workspace</h1>
<p class="notice" id="training-status"></p>
<p id="identity"></p>
<details><summary>Системная инструкция</summary><p id="system-prompt"></p></details>
<details><summary>Checkpoint bridge</summary><p id="checkpoint"></p></details>
<h2>Запрос</h2><p id="prompt"></p>
<h2>Ответ Iris</h2><p id="answer"></p>
<label for="cycle">Цикл <output id="cycle-number"></output></label>
<input id="cycle" type="range" min="0" value="0" step="1">
<p id="selection" aria-live="polite"></p>
<div class="scroll"><table><caption>Модули: заявка на запись</caption>
<thead><tr><th>Модуль</th><th>Bid</th><th>Confidence</th><th>Выбран</th></tr></thead>
<tbody id="modules"></tbody></table></div>
<div class="scroll"><table><caption>Слоты после записи</caption>
<thead><tr><th>Слот</th><th>Занят</th><th>Источник</th><th>Норма</th><th>Возраст</th><th>TTL</th></tr></thead>
<tbody id="slots"></tbody></table></div>
<h2>Latent embeddings перед генерацией</h2>
<p id="latent-summary"></p>
<div class="scroll"><table><caption>Последний цикл, независимо от ползунка выше</caption>
<thead><tr><th>Вектор</th><th>Слот</th><th>Занят</th><th>Источник</th>
<th>Позиция</th><th>Mask</th><th>L2 норма</th></tr></thead>
<tbody id="latents"></tbody></table></div>
<h2>Границы результата</h2><p id="limits"></p>
<script id="trace-data" type="application/json">DATA_PLACEHOLDER</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("trace-data").textContent);
const result=data.result, trace=data.trace;
const set=(id,value)=>{document.getElementById(id).textContent=value;};
set("identity","CPU · "+trace.num_modules+" модулей · "+trace.num_slots+
 " слотов · layout: "+(result.prefix_layout || "suffix"));
set("prompt",result.prompt);set("answer",result.text);
set("system-prompt",result.system_prompt || "Отсутствует");
const trained=result.kind==="iris-coursework-trained-bridge-demo-v1";
set("training-status",trained?
 "Учебный режим: обучен только bridge; когнитивные модули необучены. "+
 "Улучшение качества не доказано.":
 "Учебный режим: bridge и когнитивные модули необучены. "+
 "Trace показывает вычисления; улучшение качества не доказано.");
set("checkpoint",trained?"Шагов обучения: "+result.checkpoint.completed_steps+
 ". Manifest SHA256: "+result.checkpoint.manifest.sha256+
 ". State SHA256: "+result.checkpoint.state.sha256:"Отсутствует");
set("limits","Stop: "+result.stop_reason+". Seed: "+result.seed+
 ". Обучение всего sidecar: нет. Научная приёмка: нет.");
const slider=document.getElementById("cycle");
slider.max=String(trace.cycles.length-1);
const number=(v)=>Number(v).toFixed(4);
function row(parent,values){
 const tr=document.createElement("tr");
 for(const value of values){
  const td=document.createElement("td");td.textContent=String(value);tr.appendChild(td);
 }
 parent.appendChild(tr);
}
function draw(){
 const c=trace.cycles[Number(slider.value)];
 set("cycle-number",c.cycle);
 set("selection","Выбранные модули (нумерация 0-based): "+c.selected_modules.join(", ")+
  ". Слоты записи: "+c.selected_slots.join(", ")+".");
 const modules=document.getElementById("modules"), slots=document.getElementById("slots");
 modules.replaceChildren();slots.replaceChildren();
 c.module_bids.forEach((bid,i)=>row(modules,[i,number(bid),number(c.module_confidence[i]),c.selected_modules.includes(i)?"да":"—"]));
 c.slot_sources.forEach((source,i)=>row(slots,[i,c.slot_occupancy[i]?"да":"нет",source<0?"—":source,number(c.slot_norms[i]),c.slot_ages[i],number(c.slot_ttl[i])]));
}
slider.addEventListener("input",draw);draw();
const latent=result.latent_tokens;
if(latent){
 set("latent-summary",latent.num_tokens+" векторов по "+latent.embedding_dim+
  " чисел; цикл "+latent.source_cycle+". Это не словарные token IDs. "+
  "Пустые слоты тоже имеют embedding. SHA256: "+latent.embedding_sha256);
 const table=document.getElementById("latents");
 latent.slot_indices.forEach((slot,i)=>row(table,[i,slot,
  latent.slot_occupancy[i]?"да":"нет",latent.slot_sources[i]<0?"—":latent.slot_sources[i],
  latent.positions[i],latent.attention_mask[i]?"true":"false",number(latent.norms[i])]));
}else{
 set("latent-summary","Для этого сохранённого артефакта latent metadata не записаны.");
}
</script></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run / "result.json", args.run / "workspace-trace.json"]
    for path in paths:
        if path.stat().st_size > 2 * 1024**2:
            raise ValueError("Bounded demo artifact required")
    result, trace = [json.loads(path.read_text("utf-8")) for path in paths]
    page = render_report(result, trace)
    output = args.run / "viewer.html"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(str(output))


if __name__ == "__main__":
    main()
