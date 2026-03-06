import pyarrow.parquet as pq
import mpmath
from datasets import Dataset
from transformers.utils import get_json_schema
from trl import SFTTrainer
    
# 读取 Parquet 文件,dataset: Jofthomas/hermes-function-calling-thinking-V1
table = pq.read_table("train-00000-of-00001.parquet")

# 转换为 Pandas DataFrame
df = table.to_pandas()

dataset = Dataset.from_dict(df)


def find_polynomial_roots(coefficients: list[float]) -> str:
    """
    Finds the roots of a polynomial given its coefficients.

    Args:
        coefficients: A list of coefficients [c_n, ..., c_1, c_0] for c_n*x^n + ... + c_0.

    Returns:
        A string representation of the list of roots.
    """
    roots = mpmath.polyroots(coefficients)
    return str(roots)


def calculate_bessel_j(order: float, value: float) -> float:
    """
    Calculates the Bessel function of the first kind, J_v(z).

    Args:
        order: The order 'v' of the Bessel function.
        value: The value 'z' at which to evaluate the function.

    Returns:
        The result of the Bessel function J_v(z).
    """
    return mpmath.besselj(order, value)


polyroots = get_json_schema(find_polynomial_roots)
bessel = get_json_schema(calculate_bessel_j)

trainer = SFTTrainer(
    #model="Qwen3-0.6B",
    model="Qwen/Qwen2.5-32B-Instruct",
    train_dataset=dataset,
)

trainer.train()
trainer.save_model('./my_model')
