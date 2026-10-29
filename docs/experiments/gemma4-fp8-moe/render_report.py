# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Embedded CSS, HTML and JavaScript keep their native line structure.
# ruff: noqa: E501

from pathlib import Path

from markdown_it import MarkdownIt

root = Path(__file__).resolve().parent
css = """
:root{color-scheme:light;--ink:#173046;--blue:#21678a;--line:#d5e1e6}
*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1150px;margin:32px auto;padding:38px 48px;background:white;border-radius:14px;box-shadow:0 8px 36px #14324b12}
h1{font-size:32px;line-height:1.2}h2{font-size:24px;margin-top:40px;border-bottom:1px solid var(--line);padding-bottom:8px}h3{font-size:19px}a{color:#116694}code,pre{font:13px/1.55 ui-monospace,monospace}code{background:#edf2f5;padding:2px 4px;border-radius:3px}pre{background:#f3f6f8;padding:18px;border-radius:8px;overflow:auto}pre code{padding:0}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}th,td{padding:10px 12px;border:1px solid var(--line);vertical-align:top}th{background:#eaf2f6;text-align:left}li{margin:6px 0}.lab{background:#f2f7fa;border:1px solid var(--line);padding:22px;border-radius:10px;margin:26px 0}.lab h2{margin:0 0 12px;border:0}.controls{display:flex;flex-wrap:wrap;gap:8px}button{font:inherit;font-size:14px;padding:8px 14px;border:1px solid #9cb6c4;background:white;border-radius:5px;color:var(--ink);cursor:pointer}button[aria-pressed=true]{background:var(--blue);color:white}.sticks{display:flex;flex-wrap:wrap;gap:7px;margin:20px 0}.stick{height:57px;width:86px;border:1px solid #647f91;border-radius:5px;position:relative;overflow:hidden;background:#f7e5c3}.fill{height:100%;background:#b7dbea}.stick span{position:absolute;inset:0;display:grid;place-items:center;font:12px ui-monospace,monospace}.caption{font-size:14px}.flow{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:18px 0}.node{border:1px solid #a5c1ce;background:#fff;border-radius:7px;padding:12px;font-size:14px}.pill{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#527385}.foot{font-size:13px;border-top:1px solid var(--line);margin-top:35px;padding-top:15px}
@media(max-width:700px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:27px}}
@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;max-width:none}.controls{display:none}pre,table{break-inside:avoid}a{color:inherit}}
"""
lab = """<section class="lab" aria-label="Interactive stick alignment example">
<h2>Why the FP8 path needs different padding</h2>
<p>Each Gemma expert has intermediate width 704. Each hardware stick holds 128 bytes. Switch formats to see why the FP8 path pads this width:</p>
<div class="controls"><button data-mode="dl16" aria-pressed="false">DL16: 704 values</button><button data-mode="partial" aria-pressed="false">FP8: 704 values</button><button data-mode="padded" aria-pressed="true">FP8: 768 values, padded</button></div>
<div id="sticks" class="sticks"></div><p id="caption" class="caption" aria-live="polite">Six full FP8 sticks: 704 real values plus 64 zeros. The hidden width stays 2816.</p>
<div class="flow"><div class="node">Checkpoint FP8 bytes<br>Pad I: 704 → 768</div><b>→</b><div class="node">Direct QFP8WT upload<br>128 resident experts per layer</div><b>→</b><div class="node">for_each_tile<br>FP8 gate / up / down matmuls</div></div><p class="caption">Device routing selects eight experts per token. The persistent workaround visits all 128 experts and weights their contributions; short token batches are padded to 128 rows and trimmed. No tokens are packed on CPU.</p>
<p class="caption">The sticks above illustrate QFP8CH activations. Weights use QFP8WT’s 2 × 64 arrangement, built by the loader from the original FP8 codes. Gemma’s shared dense MLP needs the same treatment: K=2112 → 2176 before upload. Attention owns its FP16 or FP8 KV cache independently.</p></section>"""
js = """
const cases={dl16:{real:704,total:704,per:64,text:'11 full DL16 sticks × 64 values = 704. No intermediate-width padding is needed.'},partial:{real:704,total:704,per:128,text:'5 full FP8 sticks plus a half-filled sixth stick. The FP8 matmul path needs explicit whole-stick padding.'},padded:{real:704,total:768,per:128,text:'6 full FP8 sticks: 704 real values + 64 explicit zeros. The loader adds these zeros to gate/up outputs and down inputs; they contribute zero to the result.'}};
function show(mode){const c=cases[mode],box=document.getElementById('sticks');box.replaceChildren();for(let i=0;i<Math.ceil(c.total/c.per);i++){let filled=Math.max(0,Math.min(c.per,c.real-i*c.per));let stick=document.createElement('div');stick.className='stick';let fill=document.createElement('div');fill.className='fill';fill.style.width=(100*filled/c.per)+'%';let label=document.createElement('span');label.textContent=filled===c.per?c.per+' values':filled+(mode==='padded'?' + '+(c.per-filled)+' zero':' / '+c.per);stick.append(fill,label);box.append(stick)}document.getElementById('caption').textContent=c.text;document.querySelectorAll('button[data-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===mode?'true':'false'))}
document.querySelectorAll('button[data-mode]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.mode)));show('padded');
"""
rendered = MarkdownIt("commonmark").enable("table").render(root.joinpath("README.md").read_text())
pos = rendered.index("<h2>")
rendered = rendered[:pos] + lab + rendered[pos:]
root.joinpath("report.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma 4 FP8 MoE assessment</title><style>'
    + css
    + '</style></head><body><main><p class="pill">Spyre • experimental assessment • 29 October 2026</p>'
    + rendered
    + '<p class="foot">Self-contained HTML. Code, commands and raw results are linked alongside this report.</p></main><script>'
    + js
    + "</script></body></html>"
)
