"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[345],{8273:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>o,contentTitle:()=>d,default:()=>a,frontMatter:()=>s,metadata:()=>r,toc:()=>t});var l=i(4848),c=i(8453);const s={},d=void 0,r={id:"blogs/CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",title:"\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",description:"3.1 \u4f8b\u5b50\uff1a\u6570\u7ec4\u76f8\u52a0",source:"@site/docs/blogs/5-CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/2-CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/3-\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6.md",sourceDirName:"blogs/5-CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/2-CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0",slug:"/blogs/CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",permalink:"/docs/blogs/CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",draft:!1,unlisted:!1,editUrl:"https://buaaer-xing.github.io/docs/blogs/5-CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/2-CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/3-\u7b2c\u4e09\u7ae0\uff1a\u7b80\u5355CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6.md",tags:[],version:"current",sidebarPosition:3,frontMatter:{},sidebar:"blogs",previous:{title:"\u7b2c\u4e8c\u7ae0\uff1aCUDA\u4e2d\u7684\u7ebf\u7a0b\u7ec4\u7ec7",permalink:"/docs/blogs/CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/\u7b2c\u4e8c\u7ae0\uff1aCUDA\u4e2d\u7684\u7ebf\u7a0b\u7ec4\u7ec7"},next:{title:"\u7b2c\u56db\u7ae0\uff1aCUDA\u7a0b\u5e8f\u7684\u9519\u8bef\u68c0\u6d4b",permalink:"/docs/blogs/CUDA\u7f16\u7a0b\u5b66\u4e60\u7b14\u8bb0/CUDA\u7f16\u7a0b\u57fa\u7840\u4e0e\u5b9e\u8df5/\u9605\u8bfb\u7b14\u8bb0/\u7b2c\u56db\u7ae0\uff1aCUDA\u7a0b\u5e8f\u7684\u9519\u8bef\u68c0\u6d4b"}},o={},t=[{value:"3.1 \u4f8b\u5b50\uff1a\u6570\u7ec4\u76f8\u52a0",id:"31-\u4f8b\u5b50\u6570\u7ec4\u76f8\u52a0",level:2},{value:"3.2 CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",id:"32-cuda\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",level:2},{value:"3.2.1 \u9690\u5f62\u7684\u8bbe\u5907\u521d\u59cb\u5316",id:"321-\u9690\u5f62\u7684\u8bbe\u5907\u521d\u59cb\u5316",level:3},{value:"3.2.2 \u8bbe\u5907\u5185\u5b58\u7684\u5206\u914d\u4e0e\u91ca\u653e",id:"322-\u8bbe\u5907\u5185\u5b58\u7684\u5206\u914d\u4e0e\u91ca\u653e",level:3},{value:"3.2.3 \u4e3b\u673a\u4e0e\u8bbe\u5907\u4e4b\u95f4\u6570\u636e\u7684\u4f20\u9012",id:"323-\u4e3b\u673a\u4e0e\u8bbe\u5907\u4e4b\u95f4\u6570\u636e\u7684\u4f20\u9012",level:3},{value:"3.2.4 \u6838\u51fd\u6570\u4e2d\u6570\u636e\u4e0e\u7ebf\u7a0b\u7684\u5bf9\u5e94",id:"324-\u6838\u51fd\u6570\u4e2d\u6570\u636e\u4e0e\u7ebf\u7a0b\u7684\u5bf9\u5e94",level:3},{value:"3.2.5 \u6838\u51fd\u6570\u7684\u8981\u6c42",id:"325-\u6838\u51fd\u6570\u7684\u8981\u6c42",level:3},{value:"3.2.6 \u6838\u51fd\u6570\u4e2dif\u8bed\u53e5\u7684\u5fc5\u8981\u6027",id:"326-\u6838\u51fd\u6570\u4e2dif\u8bed\u53e5\u7684\u5fc5\u8981\u6027",level:3},{value:"3.3 \u81ea\u5b9a\u4e49\u8bbe\u5907\u51fd\u6570",id:"33-\u81ea\u5b9a\u4e49\u8bbe\u5907\u51fd\u6570",level:2},{value:"3.3.1 \u51fd\u6570\u6267\u884c\u7a7a\u95f4\u6807\u8bc6\u7b26",id:"331-\u51fd\u6570\u6267\u884c\u7a7a\u95f4\u6807\u8bc6\u7b26",level:3}];function h(e){const n={code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,c.R)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(n.h2,{id:"31-\u4f8b\u5b50\u6570\u7ec4\u76f8\u52a0",children:"3.1 \u4f8b\u5b50\uff1a\u6570\u7ec4\u76f8\u52a0"}),"\n",(0,l.jsx)(n.p,{children:"\u7565"}),"\n",(0,l.jsxs)(n.p,{children:["\u6ce8\u610f\uff1a\u5728\u5224\u65ad\u4e24\u4e2a\u6d6e\u70b9\u6570\u662f\u5426\u76f8\u7b49\u65f6\uff0c\u4e0d\u80fd\u8fd0\u7528\u8fd0\u7b97\u7b26",(0,l.jsx)(n.code,{children:"=="}),"\uff0c\u800c\u662f\u8981\u5c06\u8fd9\u4e24\u4e2a\u6570\u7684\u5dee\u7684\u7edd\u5bf9\u503c\u4e0e\u4e00\u4e2a\u5f88\u5c0f\u7684\u6570\u8fdb\u884c\u6bd4\u8f83\u3002"]}),"\n",(0,l.jsx)(n.h2,{id:"32-cuda\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6",children:"3.2 CUDA\u7a0b\u5e8f\u7684\u57fa\u672c\u6846\u67b6"}),"\n",(0,l.jsx)(n.p,{children:"\u57fa\u672c\u6846\u67b6\u5982\u4e0b\uff1a"}),"\n",(0,l.jsx)(n.pre,{children:(0,l.jsx)(n.code,{className:"language-cpp",children:"// \u5934\u6587\u4ef6\u5305\u542b\n#include <cuda_runtime.h>\n#include <iostream>\n\n// \u5e38\u91cf\u5b9a\u4e49\uff08\u6216\u8005\u5b8f\u5b9a\u4e49\uff09\n#define N 1024\n\n// C++ \u81ea\u5b9a\u4e49\u51fd\u6570\u548c CUDA \u6838\u51fd\u6570\u7684\u58f0\u660e\uff08\u539f\u578b\uff09\n__global__ void kernel_function();\n\n// int main(void)\nint main(void)\n{\n    // \u5206\u914d\u4e3b\u673a\u4e0e\u8bbe\u5907\u5185\u5b58\n    int* host_memory;\n    int* device_memory;\n    cudaMalloc((void**)&device_memory, N * sizeof(int));\n    host_memory = (int*)malloc(N * sizeof(int));\n\n    // \u521d\u59cb\u5316\u4e3b\u673a\u4e2d\u7684\u6570\u636e\n    for (int i = 0; i < N; i++) {\n        host_memory[i] = i;\n    }\n\n    // \u5c06\u67d0\u4e9b\u6570\u636e\u4ece\u4e3b\u673a\u590d\u5236\u5230\u8bbe\u5907\n    cudaMemcpy(device_memory, host_memory, N * sizeof(int), cudaMemcpyHostToDevice);\n\n    // \u8c03\u7528\u6838\u51fd\u6570\u5728\u8bbe\u5907\u4e2d\u8fdb\u884c\u8ba1\u7b97\n    kernel_function<<<1, N>>>();\n\n    // \u5c06\u67d0\u4e9b\u6570\u636e\u4ece\u8bbe\u5907\u590d\u5236\u5230\u4e3b\u673a\n    cudaMemcpy(host_memory, device_memory, N * sizeof(int), cudaMemcpyDeviceToHost);\n\n    // \u91ca\u653e\u4e3b\u673a\u4e0e\u8bbe\u5907\u5185\u5b58\n    cudaFree(device_memory);\n    free(host_memory);\n\n    return 0;\n}\n\n// C++ \u81ea\u5b9a\u4e49\u51fd\u6570\u548c CUDA \u6838\u51fd\u6570\u7684\u5b9a\u4e49\uff08\u5b9e\u73b0\uff09\n__global__ void kernel_function() {\n    // \u6838\u51fd\u6570\u7684\u5185\u5bb9\n}\n"})}),"\n",(0,l.jsx)(n.h3,{id:"321-\u9690\u5f62\u7684\u8bbe\u5907\u521d\u59cb\u5316",children:"3.2.1 \u9690\u5f62\u7684\u8bbe\u5907\u521d\u59cb\u5316"}),"\n",(0,l.jsxs)(n.p,{children:["\u5728CUDA\u8fd0\u884c\u65f6API\u4e2d\uff0c\u6ca1\u6709\u660e\u663e\u5730\u521d\u59cb\u5316\u8bbe\u5907\uff08\u5373GPU)\u7684\u51fd\u6570\u3002",(0,l.jsx)("font",{color:"red",children:(0,l.jsx)("b",{children:"\u5728\u7b2c\u4e00\u6b21\u8c03\u7528\u4e00\u4e2a\u548c\u8bbe\u5907\u7ba1\u7406\u53ca\u7248\u672c\u67e5\u8be2\u529f\u80fd\u65e0\u5173\u7684\u8fd0\u884c\u65f6API\u51fd\u6570\u65f6\uff0c\u8bbe\u5907\u5c06\u81ea\u52a8\u521d\u59cb\u5316"})}),"\u3002"]}),"\n",(0,l.jsxs)(n.ol,{children:["\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"\u81ea\u52a8\u521d\u59cb\u5316\u8bbe\u5907"}),"\uff1a\u5f53\u5728 CUDA \u7a0b\u5e8f\u4e2d\u7b2c\u4e00\u6b21\u8c03\u7528\u4e00\u4e2a\u4e0e\u8bbe\u5907\u65e0\u5173\u7684\u51fd\u6570\u65f6\uff08\u4f8b\u5982\uff1a\u5185\u5b58\u5206\u914d\u51fd\u6570 ",(0,l.jsx)(n.code,{children:"cudaMalloc()"})," \u6216\u6838\u51fd\u6570\u8c03\u7528",(0,l.jsx)(n.code,{children:" kernel<<<>>>"}),"\uff09\uff0cCUDA \u8fd0\u884c\u65f6\u4f1a\u81ea\u52a8\u8fdb\u884c\u8bbe\u5907\u7684\u521d\u59cb\u5316\u3002\u8fd9\u610f\u5473\u7740\u4f60\u65e0\u9700\u624b\u52a8\u8c03\u7528\u67d0\u4e2a\u521d\u59cb\u5316\u51fd\u6570\uff0cCUDA \u4f1a\u5728\u9700\u8981\u65f6\u81ea\u52a8\u9009\u62e9\u548c\u521d\u59cb\u5316 GPU\u3002"]}),"\n",(0,l.jsxs)(n.li,{children:[(0,l.jsx)(n.strong,{children:"\u4e0e\u8bbe\u5907\u7ba1\u7406\u53ca\u7248\u672c\u67e5\u8be2\u65e0\u5173\u7684\u51fd\u6570"}),"\uff1a\u8fd9\u91cc\u7684\u201c\u4e0e\u8bbe\u5907\u7ba1\u7406\u53ca\u7248\u672c\u67e5\u8be2\u65e0\u5173\u201d\u662f\u6307\u4e00\u4e9b CUDA API \u51fd\u6570\uff0c\u6bd4\u5982 ",(0,l.jsx)(n.code,{children:"cudaGetDevice()"})," \u6216 ",(0,l.jsx)(n.code,{children:"cudaDeviceProp"}),"\uff0c\u8fd9\u4e9b\u51fd\u6570\u662f\u4e13\u95e8\u7528\u4e8e\u67e5\u8be2\u6216\u7ba1\u7406\u8bbe\u5907\u4fe1\u606f\u7684\uff0c\u5b83\u4eec\u672c\u8eab\u5e76\u4e0d\u4f1a\u89e6\u53d1 GPU \u7684\u81ea\u52a8\u521d\u59cb\u5316\u3002\u53ea\u6709\u5f53\u4f60\u8c03\u7528\u771f\u6b63\u6d89\u53ca\u5230 GPU \u8ba1\u7b97\u6216\u5185\u5b58\u5206\u914d\u7684\u51fd\u6570\u65f6\uff0cGPU \u624d\u4f1a\u88ab\u521d\u59cb\u5316\u3002"]}),"\n"]}),"\n",(0,l.jsxs)(n.p,{children:["\u6240\u4ee5\uff0c\u5728\u8fdb\u884c\u51fd\u6570\u7684\u6d4b\u901f\u4e4b\u524d\uff0c\u4e00\u5b9a\u8981\u5148\u8fdb\u884c",(0,l.jsx)("font",{color:"red",children:(0,l.jsx)("b",{children:"warm up"})}),"\u7684\u64cd\u4f5c\uff0c\u907f\u514d\u6d4b\u901f\u7684\u8bef\u5dee\u3002"]}),"\n",(0,l.jsx)(n.h3,{id:"322-\u8bbe\u5907\u5185\u5b58\u7684\u5206\u914d\u4e0e\u91ca\u653e",children:"3.2.2 \u8bbe\u5907\u5185\u5b58\u7684\u5206\u914d\u4e0e\u91ca\u653e"}),"\n",(0,l.jsx)(n.p,{children:"\u7565"}),"\n",(0,l.jsx)(n.h3,{id:"323-\u4e3b\u673a\u4e0e\u8bbe\u5907\u4e4b\u95f4\u6570\u636e\u7684\u4f20\u9012",children:"3.2.3 \u4e3b\u673a\u4e0e\u8bbe\u5907\u4e4b\u95f4\u6570\u636e\u7684\u4f20\u9012"}),"\n",(0,l.jsx)(n.p,{children:"\u7565"}),"\n",(0,l.jsx)(n.h3,{id:"324-\u6838\u51fd\u6570\u4e2d\u6570\u636e\u4e0e\u7ebf\u7a0b\u7684\u5bf9\u5e94",children:"3.2.4 \u6838\u51fd\u6570\u4e2d\u6570\u636e\u4e0e\u7ebf\u7a0b\u7684\u5bf9\u5e94"}),"\n",(0,l.jsx)(n.p,{children:"\u7565"}),"\n",(0,l.jsx)(n.h3,{id:"325-\u6838\u51fd\u6570\u7684\u8981\u6c42",children:"3.2.5 \u6838\u51fd\u6570\u7684\u8981\u6c42"}),"\n",(0,l.jsx)(n.p,{children:"\u8fd9\u91cc\u503c\u5f97\u6ce8\u610f\u7684\u6709\u51e0\u70b9\uff1a"}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsx)(n.li,{children:"\u53ef\u4ee5\u5411\u6838\u51fd\u6570\u4f20\u9012\u975e\u6307\u9488\u53d8\u91cf\uff0c\u5176\u5185\u5bb9\u5bf9\u6bcf\u4e2a\u7ebf\u7a0b\u90fd\u53ef\u89c1\u3002"}),"\n",(0,l.jsx)(n.li,{children:"\u9664\u975e\u4f7f\u7528\u7edf\u4e00\u5185\u5b58\u7f16\u7a0b\u673a\u5236\uff0c\u5426\u5219\u4f20\u7ed9\u548c\u51fd\u6570\u7684\u6570\u7ec4\uff08\u6307\u9488\uff09\u5fc5\u987b\u6307\u5411\u8bbe\u5907\u5185\u5b58\u3002"}),"\n"]}),"\n",(0,l.jsx)(n.h3,{id:"326-\u6838\u51fd\u6570\u4e2dif\u8bed\u53e5\u7684\u5fc5\u8981\u6027",children:"3.2.6 \u6838\u51fd\u6570\u4e2dif\u8bed\u53e5\u7684\u5fc5\u8981\u6027"}),"\n",(0,l.jsx)(n.p,{children:"\u9700\u8981\u901a\u8fc7\u6761\u4ef6\u8bed\u53e5\u89c4\u907f\u4e0d\u9700\u8981\u7684\u7ebf\u7a0b\u64cd\u4f5c\u3002"}),"\n",(0,l.jsx)(n.h2,{id:"33-\u81ea\u5b9a\u4e49\u8bbe\u5907\u51fd\u6570",children:"3.3 \u81ea\u5b9a\u4e49\u8bbe\u5907\u51fd\u6570"}),"\n",(0,l.jsx)(n.p,{children:"\u6838\u51fd\u6570\u53ef\u4ee5\u8c03\u7528\u4e0d\u5e26\u6267\u884c\u914d\u7f6e\u7684\u81ea\u5b9a\u4e49\u51fd\u6570\uff0c\u8fd9\u6837\u7684\u81ea\u5b9a\u4e49\u51fd\u6570\u79f0\u4e3a\u8bbe\u5907\u51fd\u6570(device function)\u3002\u5b83\u662f\u5728\u8bbe\u5907\u4e2d\u6267\u884c\uff0c\u5e76\u5728\u8bbe\u5907\u4e2d\u88ab\u8c03\u7528\u7684\u3002\u4e0e\u4e4b\u76f8\u6bd4\uff0c\u6838\u51fd\u6570\u662f\u5728\u8bbe\u5907\u4e2d\u6267\u884c\uff0c\u4f46\u5728\u4e3b\u673a\u7aef\u88ab\u8c03\u7528\u7684\u3002"}),"\n",(0,l.jsx)(n.h3,{id:"331-\u51fd\u6570\u6267\u884c\u7a7a\u95f4\u6807\u8bc6\u7b26",children:"3.3.1 \u51fd\u6570\u6267\u884c\u7a7a\u95f4\u6807\u8bc6\u7b26"}),"\n",(0,l.jsxs)(n.ol,{children:["\n",(0,l.jsxs)(n.li,{children:["\u7528 ",(0,l.jsx)(n.code,{children:"__global__"})," \u4fee\u9970\u7684\u51fd\u6570\u79f0\u4e3a\u6838\u51fd\u6570\uff0c\u4e00\u822c\u7531\u4e3b\u673a\u8c03\u7528\uff0c\u5728\u8bbe\u5907\u4e2d\u6267\u884c\u3002\u5982\u679c\u4f7f\u7528\u52a8\u6001\u5e76\u884c\uff0c\u5219\u4e5f\u53ef\u4ee5\u5728\u6838\u51fd\u6570\u4e2d\u8c03\u7528\u81ea\u5df1\u6216\u5176\u4ed6\u6838\u51fd\u6570\u3002"]}),"\n",(0,l.jsxs)(n.li,{children:["\u7528 ",(0,l.jsx)(n.code,{children:"__device__"})," \u4fee\u9970\u7684\u51fd\u6570\u79f0\u4e3a\u8bbe\u5907\u51fd\u6570\uff0c\u53ea\u80fd\u88ab\u6838\u51fd\u6570\u6216\u5176\u4ed6\u8bbe\u5907\u51fd\u6570\u8c03\u7528\uff0c\u5728\u8bbe\u5907\u4e2d\u6267\u884c\u3002"]}),"\n",(0,l.jsxs)(n.li,{children:["\u7528 ",(0,l.jsx)(n.code,{children:"__host__"})," \u4fee\u9970\u7684\u51fd\u6570\u5c31\u662f\u4e3b\u673a\u7aef\u7684\u666e\u901a C++ \u51fd\u6570\uff0c\u5728\u4e3b\u673a\u4e2d\u88ab\u8c03\u7528\uff0c\u5728\u4e3b\u673a\u4e2d\u6267\u884c\u3002\u5bf9\u4e8e\u4e3b\u673a\u7aef\u7684\u51fd\u6570\uff0c\u8be5\u4fee\u9970\u7b26\u53ef\u7701\u7565\u3002\u4e4b\u6240\u4ee5\u63d0\u4f9b\u8fd9\u6837\u4e00\u4e2a\u4fee\u9970\u7b26\uff0c\u662f\u56e0\u4e3a\u6709\u65f6\u53ef\u4ee5\u7528 ",(0,l.jsx)(n.code,{children:"__host__"})," \u548c ",(0,l.jsx)(n.code,{children:"__device__"})," \u540c\u65f6\u4fee\u9970\u4e00\u4e2a\u51fd\u6570\uff0c\u4f7f\u5f97\u8be5\u51fd\u6570\u65e2\u662f\u4e00\u4e2a C++ \u4e2d\u7684\u666e\u901a\u51fd\u6570\uff0c\u53c8\u662f\u4e00\u4e2a\u8bbe\u5907\u51fd\u6570\u3002\u8fd9\u6837\u505a\u53ef\u4ee5\u51cf\u5c11\u5197\u4f59\u4ee3\u7801\u3002\u7f16\u8bd1\u5668\u5c06\u9488\u5bf9\u4e3b\u673a\u548c\u8bbe\u5907\u5206\u522b\u7f16\u8bd1\u8be5\u51fd\u6570\u3002"]}),"\n"]}),"\n",(0,l.jsx)(n.p,{children:"\u6ce8\u610f\uff1a\u7f16\u8bd1\u5668\u51b3\u5b9a\u628a\u8bbe\u5907\u51fd\u6570\u5f53\u4f5c\u5185\u8054\u51fd\u6570\u6216\u8005\u975e\u5185\u8054\u51fd\u6570\u3002\u4f46\u662f\uff1a"}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:["\u53ef\u4ee5\u7528\u4fee\u9970\u7b26",(0,l.jsx)(n.code,{children:"__noinline__"}),"\u5efa\u8bae\u628a\u4e00\u4e2a\u8bbe\u5907\u51fd\u6570\u4f5c\u4e3a\u975e\u5185\u8054\u51fd\u6570\uff08\u7f16\u8bd1\u5668\u4e0d\u4e00\u5b9a\u63a5\u53d7\uff09"]}),"\n",(0,l.jsxs)(n.li,{children:["\u4e5f\u53ef\u4ee5\u7528\u4fee\u9970\u7b26",(0,l.jsx)(n.code,{children:"__forceinline__"}),"\u5efa\u8bae\u4e00\u4e2a\u8bbe\u5907\u51fd\u6570\u4f5c\u4e3a\u5185\u8054\u51fd\u6570\u3002"]}),"\n"]})]})}function a(e={}){const{wrapper:n}={...(0,c.R)(),...e.components};return n?(0,l.jsx)(n,{...e,children:(0,l.jsx)(h,{...e})}):h(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>d,x:()=>r});var l=i(6540);const c={},s=l.createContext(c);function d(e){const n=l.useContext(s);return l.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(c):e.components||c:d(e.components),l.createElement(s.Provider,{value:n},e.children)}}}]);