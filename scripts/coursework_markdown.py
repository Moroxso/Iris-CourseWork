# ruff: noqa: E501
"""Shared offline-safe Markdown renderer for model responses."""

MARKDOWN_STYLE = """
.markdown-answer{line-height:1.55;overflow-wrap:anywhere}
.markdown-answer p{white-space:pre-wrap;margin:.55em 0}
.markdown-answer h1,.markdown-answer h2,.markdown-answer h3,.markdown-answer h4,.markdown-answer h5,.markdown-answer h6{margin:.9em 0 .4em;line-height:1.25}
.markdown-answer ul,.markdown-answer ol{padding-left:1.6em;margin:.5em 0}
.markdown-answer pre{white-space:pre;overflow:auto;padding:12px;border:1px solid GrayText;border-radius:6px}
.markdown-answer code{font-family:ui-monospace,Consolas,monospace}
.markdown-answer blockquote{border-left:3px solid GrayText;padding-left:12px;margin:.5em 0}
.markdown-answer a{color:LinkText}
.markdown-answer table{display:block;overflow:auto}
"""

MARKDOWN_JS = r"""
function renderMarkdown(target, source){
 source=String(source ?? "");
 if(target.dataset.markdownSource===source)return;
 target.dataset.markdownSource=source;
 target.replaceChildren();
 const lines=source.replace(/\r\n?/g,"\n").split("\n");
 const inline=(parent,value)=>{
  const pattern=/(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|\[[^\]\n]+\]\([^\s)]+\))/g;
  let pos=0,match;
  while((match=pattern.exec(value))!==null){
   parent.append(document.createTextNode(value.slice(pos,match.index)));
   const token=match[0]; let element;
   if(token[0]==="`"){element=document.createElement("code");element.textContent=token.slice(1,-1);}
   else if(token.startsWith("**")||token.startsWith("__")){element=document.createElement("strong");element.textContent=token.slice(2,-2);}
   else if(token[0]==="*"||token[0]==="_"){element=document.createElement("em");element.textContent=token.slice(1,-1);}
   else {
    const link=/^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
    try{
     const url=new URL(link[2]);
     if(!["https:","http:"].includes(url.protocol))throw new Error("Unsupported link");
     element=document.createElement("a");element.href=url.href;element.rel="noopener noreferrer";
     element.textContent=link[1];
    }catch{element=document.createTextNode(token);}
   }
   parent.append(element);pos=pattern.lastIndex;
  }
  parent.append(document.createTextNode(value.slice(pos)));
 };
 const blockStart=line=>/^\s{0,3}(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+|>|`{3,}|~{3,})/.test(line);
 let i=0;
 while(i<lines.length){
  if(!lines[i].trim()){i++;continue;}
  let match=/^\s{0,3}(`{3,}|~{3,})/.exec(lines[i]);
  if(match){
   const fence=match[1],body=[];i++;
   while(i<lines.length&&!new RegExp("^\\s{0,3}"+fence[0]+"{"+fence.length+",}\\s*$").test(lines[i]))body.push(lines[i++]);
   if(i<lines.length)i++;
   const pre=document.createElement("pre"),code=document.createElement("code");
   code.textContent=body.join("\n");pre.append(code);target.append(pre);continue;
  }
  match=/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(lines[i]);
  if(match){const h=document.createElement("h"+match[1].length);inline(h,match[2]);target.append(h);i++;continue;}
  match=/^( {0,12})([-+*]|\d+[.)])\s+(.+)$/.exec(lines[i]);
  if(match){
   const listMatch=line=>/^( {0,12})([-+*]|\d+[.)])\s+(.+)$/.exec(line);
   const parseList=(indent,ordered)=>{
    const list=document.createElement(ordered?"ol":"ul");
    if(ordered)list.start=parseInt(listMatch(lines[i])[2],10);
    while(i<lines.length){
     const item=listMatch(lines[i]);
     if(!item||item[1].length!==indent||(/^\d/.test(item[2]))!==ordered)break;
     const li=document.createElement("li");inline(li,item[3]);list.append(li);i++;
     while(i<lines.length){
      const nested=listMatch(lines[i]);
      if(!nested||nested[1].length<=indent)break;
      li.append(parseList(nested[1].length,/^\d/.test(nested[2])));
     }
    }
    return list;
   };
   target.append(parseList(match[1].length,/^\d/.test(match[2])));continue;
  }
  if(/^\s{0,3}>/.test(lines[i])){
   const quote=[];while(i<lines.length&&/^\s{0,3}>/.test(lines[i]))quote.push(lines[i++].replace(/^\s{0,3}> ?/,""));
   const block=document.createElement("blockquote");inline(block,quote.join("\n"));target.append(block);continue;
  }
  const cells=line=>line.trim().replace(/^\|/,'').replace(/\|$/,'').split("|").map(v=>v.trim());
  if(i+1<lines.length&&lines[i].includes("|")&&lines[i+1].includes("|")){
   const header=cells(lines[i]),separator=cells(lines[i+1]);
   if(header.length===separator.length&&separator.every(v=>/^:?-{3,}:?$/.test(v))){
    const table=document.createElement("table"),thead=document.createElement("thead"),tr=document.createElement("tr");
    header.forEach(value=>{const th=document.createElement("th");inline(th,value);tr.append(th);});
    thead.append(tr);table.append(thead);i+=2;
    const tbody=document.createElement("tbody");
    while(i<lines.length&&lines[i].trim()&&lines[i].includes("|")){
     const values=cells(lines[i++]),row=document.createElement("tr");
     header.forEach((_,n)=>{const td=document.createElement("td");inline(td,values[n]||"");row.append(td);});
     tbody.append(row);
    }
    table.append(tbody);target.append(table);continue;
   }
  }
  const paragraph=[lines[i++]];
  while(i<lines.length&&lines[i].trim()&&!blockStart(lines[i]))paragraph.push(lines[i++]);
  const p=document.createElement("p");inline(p,paragraph.join("\n"));target.append(p);
 }
}
"""
