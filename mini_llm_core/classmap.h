#pragma once
#include <numbers>

namespace llm
{
	using scalar = float;
	using vdb = std::vector<scalar>;

	inline scalar GELU(scalar x)
	{
		auto const static f{ static_cast<scalar>(sqrt(2. / std::numbers::pi)) };
		return  .5f * x * (1.f + static_cast<scalar>(tanh(f * (x + .044715 * x * x * x))));
	}

	class Attention;
	class Linear;
	class MultiHeadAttention;
	class Tensor;
	class RMSNorm;
	class Matrix;
	class Vector;
	template<typename AttnT>class TransformerBlock;
	//class Tokenizer;
	//class Transformer;
}
