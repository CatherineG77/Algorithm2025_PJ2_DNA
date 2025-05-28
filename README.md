# Algorithm2025_PJ2_DNA
DNA dotplot comparision(test 2 solved)
时间复杂度O（mn），空间复杂度O（mn）
grid = build_dotplot(query, reference)用B/R点标记每个位置相同或者互补的碱基对，使用图算法建立plot对角线图
然后main_ld, anti_lu = init_diag_tables(grid)建立main_ld,anti_lu两个累计对角匹配元素作为全局变量用于计算每段的错误率，使用动态规划填表方法
final_segs = find_diagonal_segments_np(grid)将grid中已经形成的连接成段的DNA片段都以numpy数组形式抽取出来，但这一步可能导致内存用量飙升，影响了test1的进行
merge_with_tolerance_np(final_segs, 1, 0.057)设置当同意对角线上段和段之间相隔一个或零个碱基（距离为1或2）的时候，如果合并段的错误率低于0.057就将段合并，并且按照从头开始的顺序贪心合并。这样可能导致全局并非最优，但通过调参降低允许错误率可以得到较优的结果。
minimal_interval_cover2_np(final_segs, 0.08, 20)通过筛选合并完成的segments然后每次选取在当前覆盖端点前的候选集中，x1 最大的段，如果和之前的已选段有重叠就进行截断，每段长度不小于20
fill_in_blanks_global(final_segs, 0.065, 25)在已选段的空隙中，通过dp表格和双端指针最后动态规划选择最优覆盖，要求是每段错误率低于0.065，长度不小于25
chose_segs(final_segs, 25)筛选选出段落中长度大于等于25的段
final_segs = extend_start_backward(final_segs, 0.1),final_segs = extend_end_backward(final_segs, 0.1)分别将每一段尽可能向前和向后延申直到错配率达到0.1或者与其他段毗邻，保证尽可能多覆盖query
最终得到结果
