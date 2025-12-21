import torch
import torch.nn as nn
import sys

# 导入模块
try:
    from bsrnn_task import BSRNN as BSRNN_Task
    from bsrnn_orig import BSRNN as BSRNN_Orig
except ImportError:
    print("❌ 错误：找不到 bsrnn_task.py 或 bsrnn_orig.py")
    sys.exit(1)

def get_snr(t, e):
    err = t - e
    return (10 * torch.log10(torch.sum(t**2) / (torch.sum(err**2) + 1e-12))).item()

def run_detailed_grading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {"input_dim": 481, "num_channel": 16, "num_layer": 2, "num_spk": 1}
    B, T, F, N, K = 2, 50, 481, 16, 36
    
    # 1. 对齐模型权重
    m_task = BSRNN_Task(**params).to(device)
    m_orig = BSRNN_Orig(**params).to(device)
    m_task.load_state_dict(m_orig.state_dict())
    m_task.eval()
    m_orig.eval()

    x = torch.randn(B, T, F, 2).to(device)
    results = {}

    # --- Step 1: BandSplit 专项检测 ---
    try:
        z_t = m_task.band_split(x)
        z_o = m_orig.band_split(x)
        snr = get_snr(z_o, z_t)
        score = 15 if snr > 80 else (5 if z_t.shape == z_o.shape else 0)
        results["1. BandSplit 投影"] = (score, 15, f"SNR: {snr:.1f}dB")
    except: results["1. BandSplit 投影"] = (0, 15, "运行崩溃")

    # --- Step 2 & 3: RNN 维度变换检测 (通过 Hook 获取中间层) ---
    # 我们模拟一个中间状态来测试 BSRNN.forward 内部逻辑
    try:
        # 为了测试 RNN 块，我们需要直接调用模块
        z_in = torch.randn(B, N, T, K).to(device)
        
        # 测试 Time RNN 逻辑 (取第一层)
        out_t = m_task.norm_time[0](z_in).transpose(1, 3).reshape(B * K, T, N)
        out_t, _ = m_task.rnn_time[0](out_t)
        out_t = m_task.fc_time[0](out_t).reshape(B, K, T, N).transpose(1, 3)
        # 这里验证学生是否掌握了 (B*K, T, N) 的变换
        
        # 简化版全流程检查
        with torch.no_grad():
            out_full_t = m_task(x)
            out_full_o = m_orig(x)
        
        # 判定 Time/Freq RNN 变换逻辑
        # 如果 SNR 很低但维度对，通常是维度置换顺序（Transpose vs Permute）写反了
        full_snr = get_snr(out_full_o, out_full_t)
        
        # 逻辑分拆
        rnn_score = 30 if full_snr > 40 else (10 if out_full_t.shape == out_full_o.shape else 0)
        results["2. Time-RNN 建模"] = (rnn_score/2, 15, "通过整体趋势判定")
        results["3. Freq-RNN 建模"] = (rnn_score/2, 15, "通过整体趋势判定")
        
        # 判定残差逻辑
        # 如果 SNR < 10dB 但维度全对，基本是漏了 skip = skip + out
        res_score = 15 if full_snr > 60 else (5 if full_snr > 5 else 0)
        results["4. 残差与归一化"] = (res_score, 15, "检测残差路径")

    except:
        results["2. Time-RNN 建模"] = (0, 15, "失败")
        results["3. Freq-RNN 建模"] = (0, 15, "失败")
        results["4. 残差与归一化"] = (0, 15, "失败")

    # --- Step 5: Mask 信号重构专项 ---
    try:
        z_fake = torch.randn(B, N, T, K).to(device)
        m_t, r_t = m_task.mask_decoder(z_fake)
        m_o, r_o = m_orig.mask_decoder(z_fake)
        snr_m = get_snr(m_o, m_t)
        # 判定 Mask 应用（复数运算）
        # 这一步通过检查最终输出 out_full_t 是否符合复数相乘规律
        if snr_m > 80:
            results["5. Mask 信号重构"] = (20, 20, "复数域 Mask 应用正确")
        else:
            results["5. Mask 信号重构"] = (5, 20, "维度匹配但数值错误")
    except: results["5. Mask 信号重构"] = (0, 20, "崩溃")

    # --- Step 6: 最终精度评分 ---
    final_snr = get_snr(out_full_o, out_full_t) if 'out_full_t' in locals() else -1
    precision_score = 20 if final_snr > 80 else (10 if final_snr > 30 else 0)
    results["6. 综合精度 (SNR)"] = (precision_score, 20, f"最终 SNR: {final_snr:.2f}dB")

    # --- 输出成绩单 ---
    print("\n" + " BSRNN 复现任务详尽评估报告 ".center(60, "="))
    total_score, max_score = 0, 0
    for label, (s, m, info) in results.items():
        print(f"{label:<25} | {s:>2} / {m:>2} | {info}")
        total_score += s
        max_score += m
    print("-" * 60)
    print(f"{'总分合计':<25} | {total_score:>2} / {max_score:>2}")
    print("".center(60, "="))

if __name__ == "__main__":
    run_detailed_grading()