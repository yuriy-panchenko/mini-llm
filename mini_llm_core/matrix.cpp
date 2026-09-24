#include "pch.h"
#include "matrix.h"

namespace llm
{
	Matrix::Matrix(size_t rows, size_t columns)
		:m_CY{ rows }
		, m_CX{ columns }
		, m_Data(rows* columns, scalar{})
	{}

	void Matrix::set(std::vector<scalar> const& data)
	{
		assert(data.size() == size());
		m_Data = data;
	}

	void Matrix::set(std::vector<scalar>&& data)
	{
		assert(data.size() == size());
		m_Data = std::move(data);
	}

	Matrix Matrix::matmul(const Matrix& oth) const
	{
		assert(m_CX == oth.m_CY);

		auto const
			rows{ m_CY },
			cols{ oth.m_CX },
			shared{ oth.m_CY };

		Matrix ret{ rows, cols };

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				for (size_t iShare = 0; iShare < shared; ++iShare)
					ret.at(iRow, iCol) += at(iRow, iShare) * oth.at(iShare, iCol);

		return ret;
	}

	Matrix Matrix::operator+(Matrix const& oth) const
	{
		auto ret{ *this };
		ret += oth;
		return ret;
	}

	void Matrix::operator+=(Matrix const& oth)
	{
		assert(size() == oth.size());
		std::transform(m_Data.begin(), m_Data.end(), oth.m_Data.begin(), m_Data.begin(), [](scalar a, scalar b) {return a + b; });
	}

	Matrix Matrix::transpose() const
	{
		Matrix ret{ m_CX, m_CY };   // dimensions swapped

		for (size_t r = 0; r < rows(); ++r)
			for (size_t c = 0; c < cols(); ++c)
				ret.at(c, r) = at(r, c);

		return ret;
	}

	Matrix Matrix::softmax() const
	{
		Matrix ret{ rows(), cols() };

		for (size_t r = 0; r < rows(); ++r)
		{
			scalar rowMax{ at(r, 0) };

			for (size_t c = 1; c < cols(); ++c)
				rowMax = std::max(rowMax, at(r, c));

			scalar sum{};
			for (size_t c = 0; c < cols(); ++c)
			{
				scalar const e{ std::exp(at(r, c) - rowMax) };
				ret.at(r, c) = e;
				sum += e;
			}

			for (size_t c = 0; c < cols(); ++c)
				ret.at(r, c) /= sum;
		}

		return ret;
	}

	Matrix Matrix::slice_cols(size_t colStart, size_t count) const
	{
		Matrix ret{ rows(), count };

		for (size_t r = 0; r < rows(); ++r)
			for (size_t c = 0; c < count; ++c)
				ret.at(r, c) = at(r, colStart + c);

		return ret;
	}

	Matrix Matrix::concat_cols(std::vector<Matrix> const& parts)
	{
		auto const rows{ parts[0].rows() };
		size_t totalCols{};
		for (auto const& p : parts)
			totalCols += p.cols();

		Matrix ret{ rows, totalCols };
		size_t colOffset{};

		for (auto const& p : parts)
		{
			auto const cols{ p.cols() };
			for (size_t r = 0; r < rows; ++r)
				for (size_t c = 0; c < cols; ++c)
					ret.at(r, colOffset + c) = p.at(r, c);
			colOffset += cols;
		}

		return ret;
	}
	Matrix Matrix::gelu() const
	{
		auto ret{ *this };
		std::transform(ret.m_Data.begin(), ret.m_Data.end(), ret.m_Data.begin(), GELU);
		return ret;
	}
}
