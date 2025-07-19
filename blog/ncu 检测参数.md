
# ncu 检测参数

```shell
/usr/local/cuda/bin/ncu --metrics lts__t_request_hit_rate,lts__t_sector_hit_rate,l1tex__t_sector_pipe_lsu_hit_rate,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active,sm__inst_executed  ./test -d 0 /home/user/congxing/BMAB_SpMV/data_link/TSOPF_RS_b2383.mtx
```

- `lts__t_request_hit_rate`
- `lts__t_sector_hit_rate`
- `l1tex__t_sector_pipe_lsu_hit_rate`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `sm__cycles_active`
- `sm__inst_executed`  