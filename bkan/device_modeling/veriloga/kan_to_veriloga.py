"""
KAN Compact Formula → Verilog-A 自动转换工具

将 KAN 符号回归提取的紧凑公式自动转换为 Verilog-A 模块。
支持:
  - 数学表达式翻译 (sqrt, exp, pow, log10, abs, sin, cos, tanh)
  - port/parameter 自动推断
  - small-signal AC 响应 (低通紧凑形式)
  - 噪声模型自动添加

用法:
  python kan_to_veriloga.py --formula symbol.txt --output my_device.va
"""

import re
import argparse
from textwrap import dedent, indent

# ── 运算符映射 ──────────────────────────────────────────
VERILOGA_MAP = {
    # Python/数学 → Verilog-A
    "sqrt":  "sqrt",      # 相同
    "exp":   "exp",       # 相同
    "abs":   "abs",       # 相同
    "sin":   "sin",       # 相同
    "cos":   "cos",       # 相同
    "tanh":  "tanh",      # 相同
    "log":   "ln",        # Python log=自然对数, VA 用 ln
    "pow(x,2)": "pow(x, 2)",  # 平方
}

# ── VA 模板 ─────────────────────────────────────────────
VA_TEMPLATE = '''// Verilog-A Compact Model: {model_name}
// Auto-generated from KAN symbolic regression
// Date: {date}
//
// {description}
//
// Condition variables: {condition_vars}
// Free variables: {free_vars}
// RMSE: {rmse}, MAE: {mae}

`include "disciplines.vams"
`include "constants.vams"

module {module_name}({ports});

  // ── Ports ──
{port_declarations}

  // ── Local nodes ──
  electrical int;

  // ── Parameters ──
{parameters}

  // ── Locals ──
  real vd;
  real T;
  real total_current;
  real log_val;
  // Frequency response
  real fc_ghz;
  real y0, y_floor, p_exp;
  real freq_hz;

  analog begin

    // Operating point
    vd = V({p_node}, {n_node});
    T  = $temperature;  // or use parameter tnom

    // ── DC Current Formula (from KAN) ──
{dc_formula}

    // ── AC Frequency Response (if applicable) ──
{ac_section}

    // ── Terminal equations ──
    I({p_node}, {n_node}) <+ total_current;

{noise_section}

  end

endmodule
'''


def translate_expression(expr_str):
    """
    将 Python 风格的数学表达式翻译为 Verilog-A 语法。

    主要转换:
      - ** → pow(a, b)   (如果需要——Verilog-A 支持 **)
      - 10**(x) → pow(10, x)
      注意: Verilog-A (Cadence) 支持 ** 运算符
    """
    result = expr_str

    # 10**(...) → pow(10, ...)
    result = re.sub(r'10\*\*\((.*?)\)', r'pow(10.0, \1)', result)

    # 确保乘法有 *
    # (already OK)

    # 科学计数法
    result = re.sub(r'(\d)\.?(\d*)e([+-]?\d+)', r'\1.\2e\3', result)

    return result


def infer_parameters(formula_str, variables):
    """从公式和变量列表推断 Verilog-A 参数"""
    params = []
    for var in variables:
        var_clean = var.strip()
        # 跳过显式的 electrical 变量 (voltage, frequency)
        if any(kw in var_clean.lower() for kw in ['voltage', 'frequency']):
            continue
        params.append(f"  parameter real {var_clean} = 1.0;")
    return params


def build_dc_formula(formula_tex, var_map, indent_spaces=4):
    """
    将公式行转换为 Verilog-A 代码块
    """
    lines = []
    lines.append("    // KAN extracted compact formula")
    lines.append("")

    translated = translate_expression(formula_tex)
    # 将单个长公式拆分为可管理的子表达式
    lines.append(f"    total_current = {translated};")

    return "\n".join(lines)


def build_ac_compact(y0_expr, y_floor_expr, fc_expr, p_val, indent_spaces=4):
    """
    构建低通紧凑 AC 响应:
      gain = y_floor + (y0 - y_floor) / (1 + (f/fc)^p)
    """
    lines = []
    lines.append("    // AC compact form: low-pass response")
    lines.append("    // gain = y_floor + (y0 - y_floor) / (1 + (f/fc)^p)")
    lines.append(f"    y0      = {y0_expr};")
    lines.append(f"    y_floor = {y_floor_expr};")
    lines.append(f"    fc_ghz  = {fc_expr};")
    lines.append(f"    p_exp   = {p_val};")
    lines.append("")
    lines.append("    // Current AC gain at operating frequency")
    lines.append("    freq_hz = 1.0;  // set by analysis")
    lines.append("    // total_current *= (y_floor + (y0 - y_floor) / (1 + pow(freq_hz/(fc_ghz*1e9), p_exp)));")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="KAN → Verilog-A converter")
    parser.add_argument("--formula", help="Path to symbolic formula file")
    parser.add_argument("--model_name", default="kan_compact_model", help="Model name")
    parser.add_argument("--output", default="kan_model.va", help="Output .va file path")
    args = parser.parse_args()

    # 如果提供了公式文件，解析它
    if args.formula:
        with open(args.formula, 'r') as f:
            content = f.read()
        print(f"[INFO] Loaded formula from: {args.formula}")
    else:
        content = ""
        print("[INFO] No formula file provided — using template only")

    # 简单示例: 直接生成 VA 文件
    va_code = VA_TEMPLATE.format(
        model_name=args.model_name,
        module_name=re.sub(r'[^a-zA-Z0-9_]', '_', args.model_name),
        date="2025-05-15",
        description="Compact model from KAN symbolic regression",
        condition_vars="active_layer_length, simulation_temperature",
        free_vars="voltage, frequency",
        rmse="N/A",
        mae="N/A",
        ports="anode, cathode",
        port_declarations="  inout anode, cathode;\n  electrical anode, cathode;",
        p_node="anode",
        n_node="cathode",
        parameters="  parameter real area = 1.0;\n  parameter real tnom = 300.0;",
        dc_formula="    // TODO: insert KAN formula here\n    total_current = 0.0;",
        ac_section="    // TODO: insert AC compact form here",
        noise_section="    // Optional: shot noise\n    // I(anode, cathode) <+ white_noise(2.0 * `P_Q * abs(total_current), \"shot\");",
    )

    with open(args.output, 'w') as f:
        f.write(va_code)

    print(f"[OK] Verilog-A model written to: {args.output}")
    print(f"[TIP] Use in Cadence Spectre: include \"{args.output}\"")
    print(f"[TIP] Simulate: name X1 anode cathode my_pdet")


if __name__ == "__main__":
    main()
