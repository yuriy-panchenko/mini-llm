#include "pch.h"
#include "tensor.h"

namespace llm
{
	Tensor::Tensor(std::initializer_list<size_t> const& dim)
		:m_Dim{ dim.begin(), dim.end() }
	{
		for (auto iter{ m_Dim.begin() }; iter != m_Dim.end(); ++iter)
			if (!*iter)
				assert(false);

		if (!m_Dim.empty())
		{
			size_t total{ 1ull };

			for (auto u : m_Dim)
				total *= u;

			m_Data.resize(total, {});
		}
	}

	Tensor Tensor::operator+(Tensor const& oth)const
	{
		auto ret{ *this };
		ret += oth;
		return ret;
	}

	void Tensor::operator+=(Tensor const& oth)
	{
		assert(size() == oth.size());
		std::transform(m_Data.begin(), m_Data.end(), oth.m_Data.begin(), m_Data.begin(), [](scalar a, scalar b) {return a + b; });
	}

	Tensor Tensor::gelu() const
	{
		auto ret{ *this };
		for (auto& val : ret.m_Data)
			val = GELU(val);
		return ret;
	}

	Tensor Tensor::norm() const
	{
		auto iter{ m_Data.begin() };
		scalar big{ std::abs(*iter) };

		for (iter++; iter != m_Data.end(); ++iter)
			big = std::max(big, std::abs(*iter));

		auto ret{ *this };

		for (auto& v : ret.m_Data)
			v /= big;

		return ret;
	}

	Tensor Tensor::matmul(const Tensor& oth) const
	{
		assert(ndim() == 2);
		assert(ndim() == oth.ndim());
		assert(shape()[1] == oth.shape()[0]);

		auto const
			rows{ shape()[0] },
			cols{ oth.shape()[1] },
			shared{ oth.shape()[0] };

		Tensor ret{ rows, cols };

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				for (size_t iShare = 0; iShare < shared; ++iShare)
					ret.at({ iRow,iCol }) += at({ iRow, iShare }) * oth.at({ iShare, iCol });

		return ret;
	}

	Tensor Tensor::softmax() const
	{
		assert(ndim() == 2ull);

		auto const rows{ shape()[0] }, cols{ shape()[1] };
		Tensor ret{ rows, cols };

		for (size_t r = 0; r < rows; ++r)
		{
			scalar rowMax{ at({ r, 0 }) };

			for (size_t c = 1; c < cols; ++c)
				rowMax = std::max(rowMax, at({ r, c }));

			scalar sum{};
			for (size_t c = 0; c < cols; ++c)
			{
				scalar const e{ std::exp(at({ r, c }) - rowMax) };
				ret.at({ r, c }) = e;
				sum += e;
			}

			for (size_t c = 0; c < cols; ++c)
				ret.at({ r, c }) /= sum;
		}

		return ret;
	}

	size_t Tensor::to_index(std::initializer_list<size_t> const& adr) const
	{
		assert(adr.size() == m_Dim.size());
		size_t ret{};

		for (size_t i = 0; i < adr.size(); ++i)
		{
			ret *= m_Dim[i];
			assert(*(adr.begin() + i) < m_Dim[i]);
			ret += *(adr.begin() + i);
		}

		assert(ret < size());

		return ret;
	}

	scalar Tensor::GELU(scalar x)
	{
		auto const static f{ static_cast<scalar>(sqrt(2. / std::numbers::pi)) };
		return  .5f * x * (1.f + static_cast<scalar>(tanh(f * (x + .044715 * x * x * x))));
	}
}
