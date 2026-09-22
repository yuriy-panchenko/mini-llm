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

	Tensor matmul(const Tensor& left, const Tensor& right)
	{
		assert(left.ndim() == 2);
		assert(left.ndim() == right.ndim());
		assert(left.shape()[1] == right.shape()[0]);

		auto const
			rows{ left.shape()[0] },
			cols{ right.shape()[1] },
			shared{ right.shape()[0] };

		Tensor ret{ rows, cols };

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				for (size_t iShare = 0; iShare < shared; ++iShare)
					ret.at({ iRow,iCol }) += left.at({ iRow, iShare }) * right.at({ iShare, iCol });

		return ret;
	}

	Tensor add(const Tensor& left, const Tensor& right)
	{
		assert(left.shape() == right.shape());

		return left + right;
	}
}
