# Algorithm2025_PJ2_DNA
DNA dotplot comparision(test 2 solved)
时间复杂度O（mn），空间复杂度O（mn）

Test2主函数如下
def find_best_matches(query: str, reference: str):
    global main_ld, grid, anti_lu
    
    grid = build_dotplot(query, reference)
    main_ld, anti_lu = init_diag_tables(grid)
    final_segs = find_diagonal_segments(grid)
    final_segs = merge_with_tolerance(final_segs, 1, 0.057)   
    final_segs = minimal_interval_cover2(final_segs, 0.08, 20)
    final_segs = fill_in_blanks_global(final_segs, 0.065, 25)
    
    
    # final_segs = weighted_interval_scheduling(final_segs)
    final_segs = chose_segs(final_segs, 25)

    final_segs = extend_start_backward(final_segs, 0.1)

    final_segs = sorted(final_segs, key=lambda s: s.x0, reverse = True)

    final_segs = extend_end_backward(final_segs, 0.1)

    final_segs = sorted(final_segs, key=lambda s: s.x0)
    
return final_segs

总时间复杂度O（mn），空间复杂度O（mn）
grid = build_dotplot(query, reference)用B/R点标记每个位置相同或者互补的碱基对，使用图算法建立plot对角线图，时间复杂度O（mn），空间复杂度O（mn）

然后main_ld, anti_lu = init_diag_tables(grid)建立main_ld,anti_lu两个累计对角匹配元素作为全局变量用于计算每段的错误率，使用动态规划填表方法，时间复杂度O（mn），空间复杂度O（mn）

final_segs = find_diagonal_segments(grid)将grid中已经形成的连接成段的DNA片段都以list形式抽取出来，通过图算法动态连接表格中每一个点和可以连接的相邻对角点，存入segment的list，时间复杂度O（mn），空间复杂度O（mn），但30k数据下空间超过100GB，导致修改调试比较困难();

merge_with_tolerance(final_segs, 1, 0.057)设置当同意对角线上段和段之间相隔一个或零个碱基（距离为1或2）的时候，如果合并段的错误率低于0.057就将段合并，并且按照从覆盖当前初始位置开始的，可覆盖到的最远位置的顺序贪心合并。这样可能导致全局并非最优，但节省时间到每条对角线的线性长度，通过调参降低允许错误率可以得到接近全局较优的结果。过程中使用桶排序，在最终的test1结果中错误率0.057被换成绝对距离的限制，作为适应10/30k样例错误率的权宜之计()时间复杂度O(mn);

minimal_interval_cover2(final_segs, 0.08, 20)通过筛选合并完成的segments然后每次选取在当前覆盖端点前的候选集中，x1 最大的段，如果和之前的已选段有重叠就进行截断，每段长度不小于20。算法也是贪心设计，同时后续对角段截断可能导致错误率增加、段变非法的风险，所以允许的错误率和合法段落长度在参数上都进一步调到更低于标准;

fill_in_blanks_global(final_segs, 0.065, 25)在已选段的空隙中，通过dp表格和双端指针最后动态规划选择最优覆盖，要求是每段错误率低于0.065，长度不小于25。段落与段落之间的空隙子区间采用find_large_valid_segments_in_range(g0, g1, rate, min_gap)函数，dp动态调度，时间复杂度是O(m’n’),累积起来不超过O(mn),然后线性合并同向小间隙段落find_large_valid_segments_in_range(g0, g1, rate, min_gap)，时间复杂度O(m)，最后weighted_interval_scheduling(filled)使用动态规划保留最优不重叠最长覆盖子段，达到尽可能填满原先段落之间空隙的效果；

chose_segs(final_segs, 25)遍历segment列表筛选选出段落中长度大于等于25的段，时间复杂度0(m),方便除去低质量小段，为之后两端延伸尽可能多匹配碱基数量腾出空间；

final_segs = extend_start_backward(final_segs, 0.1),final_segs = extend_end_backward(final_segs, 0.1)分别将每一段尽可能向前和向后延申直到错配率达到0.1或者与其他段毗邻，保证尽可能多覆盖query；

最后将segment按照x0从小到大顺序排列，排序函数python自带，时间复杂度O(mlgm)，输出的时候右端加1保证左闭右开；

最终得到结果。

Test1主函数稍有不同，要求精确性极高，而例外情况较少，所以容错（补丁）函数更少，主要靠一气呵成，调参更苛刻，内容如下：
def find_best_matches(query: str, reference: str) -> List[Segment]:
    
    global main_ld, grid, anti_lu
    """
    主函数：给定 query 和 reference，返回保留下来的所有最长匹配段
    """
    # 0. 构建 dot-plot
    showInfo("start build dot plot")
    grid = build_dotplot(query, reference)
    showInfo("build dot plot finished, start init diag tables")
    main_ld, anti_lu = init_diag_tables(grid)
    showInfo("start find diagonal segments")
    final_segs = find_diagonal_segments(grid)
showInfo("diagonal segments found")

    # # 29819!!!!! - test1 
    final_segs = merge_with_tolerance_err(final_segs, 2)
    final_segs = minimal_interval_cover_err(final_segs, 2, 30)

    final_segs = fill_in_blanks_global(final_segs, 0.03, 25)
    

    final_segs = chose_segs(final_segs, 200)

    final_segs = extend_start_backward_err(final_segs, 10)

    final_segs = sorted(final_segs, key=lambda s: s.x0, reverse = True)

    final_segs = extend_end_backward_err(final_segs, 10)

    final_segs = sorted(final_segs, key=lambda s: s.x0)
    
    
    return final_segs

函数功能与test2中函数名前缀一样的函数几乎相同，err表示原先表示rate的容错参数在此处变成了绝对错配数量max_err，时间空间复杂度依然都是O(mn)
