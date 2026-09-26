#include "pch.h"
#include "rmsnorm.h"
#include "matrix.h"

namespace llm
{
	RMSNorm::RMSNorm(Vector const& gamma, scalar eps)
		:m_Gamma{ gamma }
		, m_dGamma{ gamma.size() }
		, m_Eps{ eps }
		, m_LastInput{ 0, gamma.size() }
	{}

	Matrix RMSNorm::forward(Matrix const& input)
	{
		m_LastInput = input;
		auto const rows{ input.rows() }, cols{ input.cols() };
		Matrix ret{ rows, cols };

		for (size_t r = 0; r < rows; ++r)
		{
			scalar sumSq{};
			for (size_t c = 0; c < cols; ++c)
			{
				auto const x{ input.at(r, c) };
				sumSq += x * x;
			}

			scalar const scale{ 1.f / std::sqrt(sumSq / cols + m_Eps) };

			for (size_t c = 0; c < cols; ++c)
				ret.at(r, c) = input.at(r, c) * scale * m_Gamma[c];
		}

		return ret;
	}

	Matrix RMSNorm::backward(Matrix const& dOut)
	{
		auto const rows{ m_LastInput.rows() }, cols{ m_LastInput.cols() };
		Matrix dX{ rows, cols };

		for (size_t r = 0; r < rows; ++r)
		{
			scalar sumSq{};
			for (size_t c = 0; c < cols; ++c)
			{
				auto const x{ m_LastInput.at(r, c) };
				sumSq += x * x;
			}
			scalar const s{ 1.f / std::sqrt(sumSq / cols + m_Eps) };
			scalar const s3{ s * s * s };

			scalar g{};   // Σ_c dOut_rc * gamma_c * x_rc
			for (size_t c = 0; c < cols; ++c)
				g += dOut.at(r, c) * m_Gamma[c] * m_LastInput.at(r, c);

			for (size_t c = 0; c < cols; ++c)
			{
				auto const x{ m_LastInput.at(r, c) };
				dX.at(r, c) = s * m_Gamma[c] * dOut.at(r, c)
					- (s3 * x / cols) * g;

				m_dGamma[c] += dOut.at(r, c) * x * s;
			}
		}

		return dX;
	}

	void RMSNorm::update(scalar lr)
	{
		m_dGamma += lr;
	}

	void RMSNorm::zero_grad()
	{
		m_dGamma.fill({});
	}
}