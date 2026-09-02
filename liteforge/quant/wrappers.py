"""GPTQ / AWQ 训练后量化的薄封装（可选依赖，缺库时给出清晰指引）。

这两个库的职责是真实可部署的整数权重打包（配套反量化内核），
与 rtn.py 的"伪量化"互补：RTN 评估方法学上限，GPTQ/AWQ 产出可部署模型。
依赖安装：pip install "liteforge[quant]" 或单独安装 gptqmodel/autoawq。
"""

import logging

logger = logging.getLogger(__name__)


def quantize_gptq(model, tokenizer, calib_texts: list, bits: int = 4,
                  group_size: int = 128, sym: bool = False):
    """用 GPTQ 量化模型（原地）。返回该库的量化报告。"""
    try:
        from gptqmodel import QuantizeConfig as GPTQConfig, quantize as gptq_quantize
    except ImportError:
        try:
            from auto_gptq import AutoGPTQForCausalLM  # noqa: F401
            logger.warning("检测到旧版 auto-gptq，建议迁移到 gptqmodel")
            raise ImportError("请安装 gptqmodel: pip install gptqmodel --extra-index-url ...")
        except ImportError:
            raise ImportError("缺少 GPTQ 依赖：pip install gptqmodel")
    qcfg = GPTQConfig(bits=bits, group_size=group_size, sym=sym,
                      desc_act=False, damp_percent=0.01)
    return gptq_quantize(model, tokenizer, calib_texts, qcfg)


def quantize_awq(model_path: str, output_dir: str, calib_texts: list,
                 bits: int = 4, group_size: int = 128):
    """AWQ 独立流程（加载→量化→保存），因 awq 库自成管线，返回输出目录。"""
    try:
        from awq import AutoAWQ
    except ImportError:
        raise ImportError("缺少 AWQ 依赖：pip install autoawq")
    awq_config = {"zero_point": True, "q_group_size": group_size,
                  "w_bit": bits, "version": "GEMM"}
    model = AutoAWQ.from_pretrained(model_path)
    model.quantize(calib_texts, awq_config)
    model.save_quantized(output_dir)
    return output_dir
