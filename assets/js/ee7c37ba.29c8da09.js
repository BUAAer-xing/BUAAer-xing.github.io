"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[1425],{8809:(n,e,i)=>{i.r(e),i.d(e,{assets:()=>l,contentTitle:()=>r,default:()=>a,frontMatter:()=>o,metadata:()=>c,toc:()=>d});var s=i(5893),t=i(1151);const o={authors:{name:"BUAAer-xing",title:"\u5317\u822aHPC\u7855\u58eb\u751f",url:"https://github.com/BUAAer-xing",image_url:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/icon.png"}},r="\u5728\u4e3b\u673a\u4e2d\u5b9a\u4e49\u590d\u6742\u7ed3\u6784\u4f53\u6307\u9488\u65e0\u6cd5\u901a\u8fc7\u5185\u5b58\u62f7\u8d1d\u76f4\u63a5\u5230\u8bbe\u5907\u5185\u5b58\u4e2d",c={permalink:"/blog/\u5728\u4e3b\u673a\u4e2d\u5b9a\u4e49\u590d\u6742\u7ed3\u6784\u4f53\u6307\u9488\u65e0\u6cd5\u901a\u8fc7\u5185\u5b58\u62f7\u8d1d\u76f4\u63a5\u5230\u8bbe\u5907\u5185\u5b58\u4e2d",editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/blog/\u5728\u4e3b\u673a\u4e2d\u5b9a\u4e49\u590d\u6742\u7ed3\u6784\u4f53\u6307\u9488\u65e0\u6cd5\u901a\u8fc7\u5185\u5b58\u62f7\u8d1d\u76f4\u63a5\u5230\u8bbe\u5907\u5185\u5b58\u4e2d.md",source:"@site/blog/\u5728\u4e3b\u673a\u4e2d\u5b9a\u4e49\u590d\u6742\u7ed3\u6784\u4f53\u6307\u9488\u65e0\u6cd5\u901a\u8fc7\u5185\u5b58\u62f7\u8d1d\u76f4\u63a5\u5230\u8bbe\u5907\u5185\u5b58\u4e2d.md",title:"\u5728\u4e3b\u673a\u4e2d\u5b9a\u4e49\u590d\u6742\u7ed3\u6784\u4f53\u6307\u9488\u65e0\u6cd5\u901a\u8fc7\u5185\u5b58\u62f7\u8d1d\u76f4\u63a5\u5230\u8bbe\u5907\u5185\u5b58\u4e2d",description:"\u9519\u8bef\u4ee3\u7801",date:"2024-02-29T14:41:32.000Z",tags:[],readingTime:4.745,hasTruncateMarker:!1,authors:[{name:"BUAAer-xing",title:"\u5317\u822aHPC\u7855\u58eb\u751f",url:"https://github.com/BUAAer-xing",image_url:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/icon.png",imageURL:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/icon.png"}],frontMatter:{authors:{name:"BUAAer-xing",title:"\u5317\u822aHPC\u7855\u58eb\u751f",url:"https://github.com/BUAAer-xing",image_url:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/icon.png",imageURL:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/icon.png"}},unlisted:!1,nextItem:{title:"\u5728mac\u4e0a\u5b89\u88c5\u548c\u4f7f\u7528Lapack\u548cLapacke",permalink:"/blog/\u5728Mac\u4e0a\u4f7f\u7528Lapack"}},l={authorsImageUrls:[void 0]},d=[{value:"\u9519\u8bef\u4ee3\u7801",id:"\u9519\u8bef\u4ee3\u7801",level:2},{value:"\u9519\u8bef\u8f93\u51fa",id:"\u9519\u8bef\u8f93\u51fa",level:3},{value:"\u9519\u8bef\u539f\u56e0",id:"\u9519\u8bef\u539f\u56e0",level:3},{value:"\u89e3\u51b3\u65b9\u6848",id:"\u89e3\u51b3\u65b9\u6848",level:3},{value:"\u66f4\u6539\u540e\u7684\u4ee3\u7801",id:"\u66f4\u6539\u540e\u7684\u4ee3\u7801",level:2}];function m(n){const e={code:"code",h2:"h2",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...(0,t.a)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.h2,{id:"\u9519\u8bef\u4ee3\u7801",children:"\u9519\u8bef\u4ee3\u7801"}),"\n",(0,s.jsx)(e.p,{children:"\u7ed3\u6784\u4f53\u7684\u5b9a\u4e49\uff1a"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-cpp",children:"// \u901a\u8fc7CSR\u683c\u5f0f\u8fdb\u884c\u77e9\u9635\u5b58\u50a8\u5143\u7d20\u7684\u7ed3\u6784\u4f53\nstruct CSRmatrix\n{\n    int *row_ptr;     // \u6307\u5411\u6bcf\u4e00\u884c\u7b2c\u4e00\u4e2a\u975e\u96f6\u5143\u7d20\u5728values\u4e2d\u7684\u4f4d\u7f6e\u7684\u6570\u7ec4\n    int *col_indices; // \u6307\u5411\u6bcf\u4e2a\u975e\u96f6\u5143\u7d20\u7684\u5217\u7d22\u5f15\u7684\u6570\u7ec4\n    double *values;   // \u5b58\u50a8\u6240\u6709\u975e\u96f6\u5143\u7d20\u503c\u7684\u6570\u7ec4\n    int numRows;      // \u77e9\u9635\u7684\u884c\u6570\n    int numCols;      // \u77e9\u9635\u7684\u5217\u6570\n    int numNonzeros;  // \u975e\u96f6\u5143\u7d20\u7684\u603b\u6570\n};\n"})}),"\n",(0,s.jsx)(e.p,{children:"\u4e3b\u4f53\u4ee3\u7801\uff1a"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-cpp",children:'#include <iostream> \n#include <vector> \n#include <string> \n#include <hip/hip_runtime.h>\n\n#include "cx_utils/cx_utils.h"\n\nusing namespace std;\n\n__global__ void spmv_kernel(CSRmatrix *csrMatrix, double *vec, double *res) { \n\tint i = blockIdx.x * blockDim.x + threadIdx.x; \n\tif (i < csrMatrix->numRows) { \n\t\tres[i] = 0; \n\t\tfor (int j = csrMatrix->row_ptr[i]; j < csrMatrix->row_ptr[i + 1]; j++) {\n\t\t\t res[i] = res[i] + csrMatrix->values[j] * vec[csrMatrix->col_indices[j]]; \n\t\t }\n\t} \t\n}\n\nint main() {\n\n\tdouble current_time = get_time();\n\t\n\tstring filename = "../data/1138_bus.mtx"; // \u5047\u8bbe\u4f60\u6709\u4e00\u4e2a\u540d\u4e3a matrix.mtx \u7684\u6587\u4ef6\n\tvector<COOmatrix> coo_mtx;\n\tint numRows, numCols, nonzeros;\n\t\n\tif (read_mtx(filename, coo_mtx, numRows, numCols, nonzeros))\n\t{\n\t    cout << "\u77e9\u9635\u8bfb\u53d6\u6210\u529f\uff01" << endl;\n\t    cout << "\u77e9\u9635\u5c3a\u5bf8: " << numRows << " x " << numCols << ", \u975e\u96f6\u5143\u7d20\u6570\u91cf: " << nonzeros << endl;\n\t\n\t    CSRmatrix csr_mtx = convert_COO_to_CSR(coo_mtx, numRows, numCols);\n\t\n\t    double h_vec[numCols];\n\t    double h_res[numRows];\n\t    double *d_vec, *d_res;\n\t    CSRmatrix *d_csr;\n\t\n\t    for (int i = 0; i < numCols; i++)\n\t    {\n\t        h_vec[i] = 1.0;\n\t    }\n\t\n\t    hipMalloc(&d_vec, numCols * sizeof(double));\n\t    hipMalloc(&d_res, numRows * sizeof(double));\n\t    hipMalloc(&d_csr, sizeof(CSRmatrix));\n\t    hipMemcpy(d_vec, h_vec, numCols * sizeof(double), hipMemcpyHostToDevice);\n\t    hipMemcpy(d_csr, &csr_mtx, sizeof(CSRmatrix), hipMemcpyHostToDevice);\n\t\n\t    int blockSize = 256;\n\t    int gridSize = (numRows + blockSize - 1) / blockSize;\n\t\n\t    spmv_kernel<<<gridSize, blockSize>>>(d_csr, d_vec, d_res);\n\t\n\t    hipDeviceSynchronize();\n\t\n\t    hipMemcpy(h_res, d_res, numRows * sizeof(double), hipMemcpyDeviceToHost);\n\t\n\t    for (int i = 0; i < numRows; i++)\n\t    {\n\t        cout << h_res[i] << endl;\n\t    }\n\t\n\t    hipFree(d_vec);\n\t    hipFree(d_res);\n\t}\n\treturn 0;\n}\n'})}),"\n",(0,s.jsx)(e.h3,{id:"\u9519\u8bef\u8f93\u51fa",children:"\u9519\u8bef\u8f93\u51fa"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-shell",children:"Invalid address access: 0x7ffc45259000, Error code: 1. \nAborted\n"})}),"\n",(0,s.jsx)(e.h3,{id:"\u9519\u8bef\u539f\u56e0",children:"\u9519\u8bef\u539f\u56e0"}),"\n",(0,s.jsxs)(e.p,{children:["\u5728CUDA\u6216HIP\u7f16\u7a0b\u4e2d\uff0c\u9047\u5230\u65e0\u6548\u5730\u5740\u9519\u8bef\u901a\u5e38\u662f\u56e0\u4e3a",(0,s.jsx)(e.strong,{children:"\u8bd5\u56fe\u8bbf\u95ee\u672a\u6b63\u786e\u5206\u914d\u6216\u4f20\u9012\u5230\u8bbe\u5907\uff08GPU\uff09\u7684\u5185\u5b58"}),"\u3002\u95ee\u9898\u5f88\u53ef\u80fd\u51fa\u73b0\u5728\u5982\u4f55\u5904\u7406",(0,s.jsx)(e.code,{children:"CSRmatrix"}),"\u7ed3\u6784\u53ca\u5176\u6210\u5458\u7684\u5185\u5b58\u5206\u914d\u548c\u4f20\u9012\u4e0a\u3002"]}),"\n",(0,s.jsxs)(e.ol,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"CSRmatrix"}),"\u7ed3\u6784\u4f53\u7684\u5b9a\u4e49\u4e2d\u5305\u542b\u56db\u4e2a\u4e3b\u8981\u6210\u5458\uff1a",(0,s.jsx)(e.code,{children:"numRows"}),"\uff08\u77e9\u9635\u7684\u884c\u6570\uff09\uff0c",(0,s.jsx)(e.code,{children:"row_ptr"}),"\uff08\u884c\u6307\u9488\u6570\u7ec4\uff09\uff0c",(0,s.jsx)(e.code,{children:"values"}),"\uff08\u975e\u96f6\u503c\u6570\u7ec4\uff09\uff0c\u548c",(0,s.jsx)(e.code,{children:"col_indices"}),"\uff08\u5217\u7d22\u5f15\u6570\u7ec4\uff09\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:["\u5f53\u6211\u4eec\u5c06",(0,s.jsx)(e.code,{children:"CSRmatrix"}),"\u7684\u4e00\u4e2a\u5b9e\u4f8b",(0,s.jsx)(e.code,{children:"csr_mtx"}),"\u4ece\u4e3b\u673a\u5185\u5b58\u590d\u5236\u5230\u4e86\u8bbe\u5907\u5185\u5b58\uff08",(0,s.jsx)(e.code,{children:"d_csr"}),"\uff09\u4e2d\u65f6\uff0c",(0,s.jsx)(e.code,{children:"CSRmatrix"}),"\u5185\u90e8\u7684\u6307\u9488\uff08\u5982",(0,s.jsx)(e.code,{children:"row_ptr"}),"\uff0c",(0,s.jsx)(e.code,{children:"values"}),"\u548c",(0,s.jsx)(e.code,{children:"col_indices"}),"\uff09\u6307\u5411\u7684\u6570\u636e\u5b9e\u9645\u4e0a\u8fd8\u5728\u4e3b\u673a\u5185\u5b58\u4e2d\uff08\u4e5f\u5c31\u662f\u8bf4\uff0c\u5728\u62f7\u8d1d\u65f6\uff0c\u53ea\u4f1a\u62f7\u8d1d\u7ed3\u6784\u4f53\u4e2d\u7684\u5177\u4f53\u6570\u636e\uff0c\u5bf9\u7ed3\u6784\u4f53\u4e2d\u6307\u9488\u6307\u5411\u7684\u6570\u636e\u5e76\u4e0d\u4f1a\u8fdb\u884c\u62f7\u8d1d\uff09\u3002\u8fd9\u610f\u5473\u7740\uff0c\u5f53GPU\u8bd5\u56fe\u901a\u8fc7\u8fd9\u4e9b\u6307\u9488\u8bbf\u95ee\u6570\u636e\u65f6\uff0c\u4f1a\u9047\u5230\u65e0\u6548\u5730\u5740\u9519\u8bef\uff0c\u56e0\u4e3a\u8fd9\u4e9b\u6307\u9488\u5bf9\u8bbe\u5907\u6765\u8bf4\u6ca1\u6709\u610f\u4e49\u3002"]}),"\n"]}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240229223613.png",alt:"image.png|center|600"})}),"\n",(0,s.jsx)(e.h3,{id:"\u89e3\u51b3\u65b9\u6848",children:"\u89e3\u51b3\u65b9\u6848"}),"\n",(0,s.jsxs)(e.ol,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsxs)(e.strong,{children:["\u5355\u72ec\u4e3a",(0,s.jsx)(e.code,{children:"CSRmatrix"}),"\u4e2d\u7684\u6bcf\u4e2a\u6307\u9488\u6210\u5458\u5206\u914d\u8bbe\u5907\u5185\u5b58"]}),"\u3002\u8fd9\u5305\u62ec",(0,s.jsx)(e.code,{children:"row_ptr"}),"\uff0c",(0,s.jsx)(e.code,{children:"values"}),"\uff0c\u548c",(0,s.jsx)(e.code,{children:"col_indices"}),"\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"\u5c06\u8fd9\u4e9b\u6570\u7ec4\u4ece\u4e3b\u673a\u590d\u5236\u5230\u8bbe\u5907\u5185\u5b58"}),"\u3002"]}),"\n"]}),"\n",(0,s.jsx)(e.h2,{id:"\u66f4\u6539\u540e\u7684\u4ee3\u7801",children:"\u66f4\u6539\u540e\u7684\u4ee3\u7801"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-cpp",children:'#include <iostream>\n#include <vector>\n#include <string>\n#include <hip/hip_runtime.h>\n\n#include "cx_utils/cx_utils.h"\n\nusing namespace std;\n\n__global__ void spmv_kernel(int *row_ptr, int *col_indices, double *values, double *vec, double *res, int *numRows)\n{\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < *numRows)\n    {\n        double sum = 0.0;\n        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)\n        {\n            sum += values[j] * vec[col_indices[j]];\n        }\n        res[i] = sum;\n    }\n}\n\nint main()\n{\n\n    double current_time = get_time();\n\n    string filename = "../data/1138_bus.mtx"; // \u5047\u8bbe\u4f60\u6709\u4e00\u4e2a\u540d\u4e3a matrix.mtx \u7684\u6587\u4ef6\n    vector<COOmatrix> coo_mtx;\n    int numRows, numCols, nonzeros;\n\n    if (read_mtx(filename, coo_mtx, numRows, numCols, nonzeros))\n    {\n        cout << "\u77e9\u9635\u8bfb\u53d6\u6210\u529f\uff01" << endl;\n        cout << "\u77e9\u9635\u5c3a\u5bf8: " << numRows << " x " << numCols << ", \u975e\u96f6\u5143\u7d20\u6570\u91cf: " << nonzeros << endl;\n\n        CSRmatrix csr_mtx = convert_COO_to_CSR(coo_mtx, numRows, numCols);\n\n        double h_vec[numCols];\n        double h_res[numRows];\n        double *d_vec, *d_res;\n        int *d_row_ptr, *d_col_indices, *d_numRows;\n        double *d_values;\n\n        for (int i = 0; i < numCols; i++)\n        {\n            h_vec[i] = 1.0;\n        }\n\n        hipMalloc(&d_vec, numCols * sizeof(double));\n        hipMalloc(&d_res, numRows * sizeof(double));\n        hipMalloc(&d_row_ptr, (numRows + 1) * sizeof(int));\n        hipMalloc(&d_col_indices, nonzeros * sizeof(int));\n        hipMalloc(&d_values, nonzeros * sizeof(double));\n        hipMalloc(&d_numRows, sizeof(int));\n\n        hipMemcpy(d_vec, h_vec, numCols * sizeof(double), hipMemcpyHostToDevice);\n        hipMemcpy(d_row_ptr, csr_mtx.row_ptr, (numRows + 1) * sizeof(int), hipMemcpyHostToDevice);\n        hipMemcpy(d_col_indices, csr_mtx.col_indices, nonzeros * sizeof(int), hipMemcpyHostToDevice);\n        hipMemcpy(d_values, csr_mtx.values, nonzeros * sizeof(double), hipMemcpyHostToDevice);\n        hipMemcpy(d_numRows, &numRows, sizeof(int), hipMemcpyHostToDevice);\n        \n        int blockSize = 256;\n        int gridSize = (numRows + blockSize - 1) / blockSize;\n\n        spmv_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_indices, d_values, d_vec, d_res, d_numRows);\n\n        hipDeviceSynchronize();\n\n        hipMemcpy(h_res, d_res, numRows * sizeof(double), hipMemcpyDeviceToHost);\n\n        for (int i = 0; i < numRows; i++)\n        {\n            cout << h_res[i] << endl;\n        }\n\n        hipFree(d_vec);\n        hipFree(d_res);\n    }\n    return 0;\n}\n\n\n'})})]})}function a(n={}){const{wrapper:e}={...(0,t.a)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(m,{...n})}):m(n)}},1151:(n,e,i)=>{i.d(e,{Z:()=>c,a:()=>r});var s=i(7294);const t={},o=s.createContext(t);function r(n){const e=s.useContext(o);return s.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:r(n.components),s.createElement(o.Provider,{value:e},n.children)}}}]);