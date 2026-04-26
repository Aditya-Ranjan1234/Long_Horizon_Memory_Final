# Long-Horizon Memory: model comparison (run_20260426_091458)

- Episodes: **2** from `episodes_grpo_long.json`
- Memory capacity: **16**
- Decoding: greedy, max_new_tokens=32

## Summary table

```
------------------------------
                     base_1.5b
------------------------------
model                base_1.5b
n_eps                        2
n_steps                    136
parse_ok %               100.0
add %                     87.5
remove %                  12.5
noop %                     0.0
rem-correct %             70.6
mean step rwd           -0.141
mean final F1            0.113
mean precision           0.129
mean recall              0.100
mean max fill             16.0
long-horiz %              83.8
mem-full %                58.8
------------------------------
```

## Failure modes per model

### base_1.5b

**Env errors** (after parse, action was illegal):

- `memory_capacity_reached` (showing 5):
  - ep=355 step=23 op=add remove_index=None mem_fill=16
  - ep=355 step=24 op=add remove_index=None mem_fill=16
  - ep=355 step=25 op=add remove_index=None mem_fill=16
  - ep=355 step=26 op=add remove_index=None mem_fill=16
  - ep=146 step=27 op=add remove_index=None mem_fill=16
