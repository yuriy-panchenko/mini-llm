#pragma once
#include <random>
#include "classmap.h"

namespace llm
{
	class Matrix
	{
	public:
		template<typename T>
		class Row
		{
			T* b, * e;
		public:
			Row(T* p, size_t cx) :b{ p }, e{ p + cx } {}
			T* begin() { return b; }
			T* begin()const { return b; }
			T* end() { return e; }
			T* end()const { return e; }
			T operator[](size_t i)const { return *(b + i); }
			T& operator[](size_t i) { return *(b + i); }
		};

		using MatmulGrads = std::pair<Matrix, Matrix>;

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
		void operator+=(scalar);
		Matrix transpose() const;
		Matrix operator/(scalar s)const { auto ret{ *this }; ret /= s; return ret; }
		void operator/=(scalar s) { std::transform(m_Data.begin(), m_Data.end(), m_Data.begin(), [&](scalar val) {return val / s; }); }
		Matrix softmax() const;
		Matrix slice_cols(size_t colStart, size_t count) const;
		static Matrix concat_cols(std::vector<Matrix> const& parts);
		Matrix gelu()const;
		Row<scalar const> row(size_t index)const { return { m_Data.data() + index * m_CX, m_CX }; }
		Row<scalar> row(size_t index) { return { m_Data.data() + index * m_CX, m_CX }; }
		Matrix xavier(std::mt19937& rng, std::uniform_real_distribution<scalar>& dist)const;

		// dC = d(A @ B)  →  dA = dC @ Bᵀ ,  dB = Aᵀ @ dC
		static MatmulGrads d_matmul(Matrix const& A, Matrix const& B, Matrix const& dC);
		// element-wise
		Matrix d_gelu(Matrix const& dy)const;   // dy * gelu'(x)
		static Matrix d_softmax(Matrix const& softmax_out, Matrix const& dy);  // classic Jacobian: dx = s ⊙ (dy − Σ(dy⊙s))
		void fill(scalar);

	private:
		size_t to_index(size_t y, size_t x)const { return y * m_CX + x; }

		std::vector<scalar> m_Data;
		size_t m_CY, m_CX;
	};
}