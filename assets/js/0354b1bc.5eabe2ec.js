"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[3073],{6992:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>c,contentTitle:()=>i,default:()=>p,frontMatter:()=>o,metadata:()=>l,toc:()=>d});var t=r(4848),s=r(8453);const o={},i=void 0,l={id:"paper_notes/Kernel/HiCOO/Z-Morton\u987a\u5e8f",title:"Z-Morton\u987a\u5e8f",description:"\u83ab\u987f\u7801",source:"@site/docs/paper_notes/3_Kernel/HiCOO/Z-Morton\u987a\u5e8f.md",sourceDirName:"paper_notes/3_Kernel/HiCOO",slug:"/paper_notes/Kernel/HiCOO/Z-Morton\u987a\u5e8f",permalink:"/docs/paper_notes/Kernel/HiCOO/Z-Morton\u987a\u5e8f",draft:!1,unlisted:!1,editUrl:"https://buaaer-xing.github.io/docs/paper_notes/3_Kernel/HiCOO/Z-Morton\u987a\u5e8f.md",tags:[],version:"current",frontMatter:{},sidebar:"paper_notes",previous:{title:"\u9605\u8bfb\u7b14\u8bb0",permalink:"/docs/paper_notes/Kernel/HPC\u52a0\u901f\u4f53\u7cfb\u7ed3\u6784\u4e2dLinpack\u4f18\u5316/\u9605\u8bfb\u7b14\u8bb0"},next:{title:"\u5f20\u91cf\u7684\u7b80\u8981\u4ecb\u7ecd",permalink:"/docs/paper_notes/Kernel/HiCOO/\u5f20\u91cf\u7684\u7b80\u8981\u4ecb\u7ecd"}},c={},d=[{value:"\u83ab\u987f\u7801",id:"\u83ab\u987f\u7801",level:2},{value:"\u6982\u8ff0",id:"\u6982\u8ff0",level:3},{value:"\u7f16\u7801\u89c4\u5219",id:"\u7f16\u7801\u89c4\u5219",level:3},{value:"\u66f4\u52a0\u9ad8\u7eac\u5ea6\u7684\u7a7a\u95f4",id:"\u66f4\u52a0\u9ad8\u7eac\u5ea6\u7684\u7a7a\u95f4",level:3},{value:"\u610f\u4e49",id:"\u610f\u4e49",level:3}];function a(e){const n={h2:"h2",h3:"h3",img:"img",p:"p",strong:"strong",...(0,s.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h2,{id:"\u83ab\u987f\u7801",children:"\u83ab\u987f\u7801"}),"\n",(0,t.jsx)(n.h3,{id:"\u6982\u8ff0",children:"\u6982\u8ff0"}),"\n",(0,t.jsxs)(n.p,{children:["\u83ab\u987f\u7801\u662f",(0,t.jsx)(n.strong,{children:"\u5c06\u591a\u7ef4\u6570\u636e\u8f6c\u5316\u4e3a\u4e00\u7ef4\u6570\u636e\u7684\u7f16\u7801"}),"\u3002"]}),"\n",(0,t.jsxs)(n.p,{children:["\u83ab\u987f\u7f16\u7801\u5b9a\u4e49\u4e86",(0,t.jsx)(n.strong,{children:"\u4e00\u6761 Z \u5f62\u7684\u7a7a\u95f4\u586b\u5145\u66f2\u7ebf"}),"\uff0c\u56e0\u6b64\u83ab\u987f\u7f16\u7801\u901a\u5e38\u4e5f\u79f0Z\u9636\u66f2\u7ebf(Z-order curve)\u3002 \u5728 N \u7ef4\u7a7a\u95f4\u4e2d\u5bf9\u4e8e\u5f7c\u6b64\u63a5\u8fd1\u7684\u5750\u6807\u5177\u6709\u5f7c\u6b64\u63a5\u8fd1\u7684\u83ab\u987f\u7801, \u53ef\u4ee5\u5e94\u7528\u4e8e\u4e3a\u4e00\u4e2a\u6574\u6570\u5bf9\u4ea7\u751f\u4e00\u4e2a\u552f\u4e00\u7d22\u5f15\u3002\u4f8b\u5982\uff0c\u5bf9\u4e8e\u5750\u6807\u7cfb\u4e2d\u7684\u5750\u6807\u70b9\u4f7f\u7528\u83ab\u987f\u7f16\u7801\u751f\u6210\u7684\u83ab\u987f\u7801\uff0c\u53ef\u4ee5\u552f\u4e00\u7d22\u5f15\u5bf9\u5e94\u7684\u70b9\u3002\u8fd9\u4e9b\u7d22\u5f15\u4e3a\u201cZ\u201d\u5f62\u6392\u5e8f \u3002\u5982\u4e0b\u56fe\u4ee5Z\u5f62(\u5de6\u4e0a->\u53f3\u4e0a->\u5de6\u4e0b->\u53f3\u4e0b)\u5206\u522b\u4ee3\u88681*1\u30012*2\u30014*4\u30018*8 \u5e73\u65b9\u5355\u4f4d\uff1a\n",(0,t.jsx)(n.img,{src:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181235.png",alt:"image.png|center|600"})]}),"\n",(0,t.jsx)(n.h3,{id:"\u7f16\u7801\u89c4\u5219",children:"\u7f16\u7801\u89c4\u5219"}),"\n",(0,t.jsx)(n.p,{children:"\u5341\u8fdb\u5236\u7f16\u7801\u89c4\u5219\uff1a\u9996\u5148\uff0c\u884c\u5217\u53f7\u8f6c\u4e3a\u4e8c\u8fdb\u5236\uff08\u4ece\u7b2c0\u884c0\u5217\u5f00\u59cb\uff09\uff1b\u7136\u540e\u884c\u5217\u53f7\u4ea4\u53c9\u6392\u5217\uff08yxyx\u2026\uff09\uff1b\u6700\u540e\u5c06\u4e8c\u8fdb\u5236\u7ed3\u679c\u8f6c\u4e3a\u5341\u8fdb\u5236\u3002Morton\u7f16\u7801\u662f\u6309\u5de6\u4e0a\uff0c\u53f3\u4e0a\uff0c\u5de6\u4e0b\uff0c\u53f3\u4e0b\u7684\u987a\u5e8f\u4ece0\u5f00\u59cb\u5bf9\u6bcf\u4e2a\u683c\u7f51\u8fdb\u884c\u81ea\u7136\u7f16\u7801\u7684\u3002\u5982\u4e0b\u56fe\uff08\u4e8c\u7ef4\u7a7a\u95f4\uff09\uff1a\u5c55\u793a\u4e868*8\u7684\u56fe\u50cf\u6bcf\u4e2a\u50cf\u7d20\u7684\u7a7a\u95f4\u7f16\u7801\uff0c\u4ece000000\u5230111111\uff0c\u7528\u4e00\u7ef4\u4e8c\u8fdb\u5236\u6570\uff0c\u7f16\u7801\u4e86x,y\u503c\u57280-7\u7684\u4f4d\u7f6e\u5750\u6807\u3002\u56fe\u4e2d\u84dd\u8272\u6570\u5b57\u4ee3\u8868x\u8f74\uff0c\u7ea2\u8272\u6570\u5b57\u4ee3\u8868y\u8f74\uff0c\u7f51\u683c\u4e2d\u7684\u4e8c\u8fdb\u5236\u6570\u7531x\u548cy\u7684\u4e8c\u8fdb\u5236\u6570\u4ea4\u53c9\u6784\u6210\u3002"}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{src:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181417.png",alt:"image.png|center|600"})}),"\n",(0,t.jsx)(n.h3,{id:"\u66f4\u52a0\u9ad8\u7eac\u5ea6\u7684\u7a7a\u95f4",children:"\u66f4\u52a0\u9ad8\u7eac\u5ea6\u7684\u7a7a\u95f4"}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{src:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181459.png",alt:"image.png|center|600"})}),"\n",(0,t.jsx)(n.h3,{id:"\u610f\u4e49",children:"\u610f\u4e49"}),"\n",(0,t.jsxs)(n.p,{children:["Z-Morton\u66f2\u7ebf\u7684\u547d\u540d\u6765\u81ea\u4e8eMorton\u7801\uff0c\u4e5f\u88ab\u79f0\u4e3a",(0,t.jsx)(n.strong,{children:"Z-order\u7801"}),"\u3002"]}),"\n",(0,t.jsxs)(n.p,{children:["Morton\u7801\u662f\u4e00\u79cd\u5c06\u591a\u7ef4\u7a7a\u95f4\u4e2d\u7684\u5750\u6807\u6620\u5c04\u5230\u4e00\u7ef4\u7a7a\u95f4\u7684\u65b9\u6cd5\uff0c\u5b83\u4e0eZ-Morton\u66f2\u7ebf\u6709\u5173\u8054\u3002\u5728Morton\u7801\u4e2d\uff0c",(0,t.jsx)("font",{color:"red",children:(0,t.jsx)("b",{children:"\u76f8\u90bb\u7684\u591a\u7ef4\u5750\u6807\u88ab\u7f16\u7801\u4e3a\u76f8\u90bb\u7684\u4e00\u7ef4\u7d22\u5f15"})}),"\uff0c\u8fd9\u79cd\u7f16\u7801\u65b9\u5f0f\u4f7f\u5f97",(0,t.jsx)("font",{color:"red",children:(0,t.jsx)("b",{children:"\u5728\u591a\u7ef4\u7a7a\u95f4\u4e2d\u7684\u6570\u636e\u80fd\u591f\u4ee5\u7ebf\u6027\u65b9\u5f0f\u8fdb\u884c\u5b58\u50a8\u548c\u8bbf\u95ee"})}),"\u3002"]}),"\n",(0,t.jsxs)(n.p,{children:["Z-Morton\u66f2\u7ebf\u548cMorton\u7801\u7684\u7279\u70b9\u4f7f\u5f97\u5b83\u4eec\u5728\u7a7a\u95f4\u7d22\u5f15\u6570\u636e\u7ed3\u6784\uff08\u6bd4\u5982\u56db\u53c9\u6811\u3001\u516b\u53c9\u6811\u7b49\uff09\u3001\u5e76\u884c\u8ba1\u7b97\u3001\u4ee5\u53ca\u4e00\u4e9b\u8ba1\u7b97\u51e0\u4f55\u7b97\u6cd5\u4e2d\u6709\u7740\u5e7f\u6cdb\u7684\u5e94\u7528\uff0c\u56e0\u4e3a",(0,t.jsx)(n.strong,{children:"\u5b83\u4eec\u80fd\u591f\u6709\u6548\u5730\u5904\u7406\u591a\u7ef4\u6570\u636e\uff0c\u5e76\u63d0\u4f9b\u9ad8\u6548\u7684\u8bbf\u95ee\u65b9\u5f0f"}),"\u3002"]})]})}function p(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(a,{...e})}):a(e)}},8453:(e,n,r)=>{r.d(n,{R:()=>i,x:()=>l});var t=r(6540);const s={},o=t.createContext(s);function i(e){const n=t.useContext(o);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:i(e.components),t.createElement(o.Provider,{value:n},e.children)}}}]);