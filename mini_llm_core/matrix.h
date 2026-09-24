#pragma once
#include <vector>
#include <algorithm>
#include "classmap.h"
namespace llm
{
	class Matrix
	{
		std::vector<scalar> m_Data;
		size_t m_CY, m_CX;
	public:
		Matrix(size_t rows, size_t columns);

		size_t size()const { return m_Data.size(); }

		void set(std::vector<scalar>const& data);
		void set(std::vector<scalar>&& data);

		scalar at(size_t row, size_t col)const { return m_Data[to_index(row, col)]; }
		scalar& at(size_t row, size_t col) { return m_Data[to_index(row, col)]; }

		Matrix matmul(const Matrix& oth) const;
		size_t rows()const { return m_CY; }
		size_t cols()const { return m_CX; }
		Matrix operator+(Matrix const&)const;
		void operator+=(Matrix const&);
		Matrix transpose() const;
		Matrix operator/(scalar s)const { auto ret{ *this }; ret /= s; return ret; }
		void operator/=(scalar s) { std::transform(m_Data.begin(), m_Data.end(), m_Data.begin(), [&](scalar val) {return val / s; }); }
		Matrix softmax() const;
		Matrix slice_cols(size_t colStart, size_t count) const;
		static Matrix concat_cols(std::vector<Matrix> const& parts);
		Matrix gelu()const;

	private:
		//scalar operator[](size_t index)const { return m_Data[index]; }
		//scalar& operator[](size_t index) { return m_Data[index]; }

		size_t to_index(size_t y, size_t x)const { return y * m_CX + x; }
	};
}