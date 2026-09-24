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

	Tensor Tensor::operator*(scalar s) const
	{
		auto ret{ *this };
		ret *= s;
		return ret;
	}

	void Tensor::operator*=(scalar s)
	{
		for (auto& val : m_Data)
			val *= s;
	}

	Tensor Tensor::operator/(scalar s) const
	{
		auto ret{ *this };
		ret /= s;
		return ret;
	}

	void Tensor::operator/=(scalar s)
	{
		for (auto& val : m_Data)
			val /= s;
	}

	Tensor Tensor::gelu() const
	{
		auto ret{ *this };
		std::transform(ret.m_Data.begin(), ret.m_Data.end(), ret.m_Data.begin(), GELU);
		//for (auto& val : ret.m_Data)
		//	val = GELU(val);
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

	/*Tensor Tensor::slice_cols(size_t colStart, size_t count) const
	{
		auto const rows{ shape()[0] };
		Tensor ret{ rows, count };

		for (size_t r = 0; r < rows; ++r)
			for (size_t c = 0; c < count; ++c)
				ret.at({ r, c }) = at({ r, colStart + c });

		return ret;
	}*/

	/*Tensor Tensor::concat_cols(std::vector<Tensor> const& parts)
	{
		auto const rows{ parts[0].shape()[0] };
		size_t totalCols{};
		for (auto const& p : parts)
			totalCols += p.shape()[1];

		Tensor ret{ rows, totalCols };
		size_t colOffset{};

		for (auto const& p : parts)
		{
			auto const cols{ p.shape()[1] };
			for (size_t r = 0; r < rows; ++r)
				for (size_t c = 0; c < cols; ++c)
					ret.at({ r, colOffset + c }) = p.at({ r, c });
			colOffset += cols;
		}

		return ret;
	}*/

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
}
