#include "pch.h"
#include "linear.h"

namespace llm
{
	Linear::Linear(Matrix const& w, Vector const& bias)
		:m_Ws{ w }
		, m_dWs{ w.rows(), w.cols() }
		, m_LastInput{ 0, w.cols() }
		, m_Bias{ bias }
		, m_dBias{ bias.size() }
	{}

	Matrix Linear::forward(const Matrix& input)
	{
		m_LastInput = input;

		auto ret{ input.matmul(m_Ws) };
		auto const
			rows{ ret.rows() },
			cols{ ret.cols() };
		assert(m_Bias.size() == cols);

		for (size_t iRow = 0; iRow < rows; ++iRow)
		{
			auto itBias{ m_Bias.begin() };

			for (auto& val : ret.row(iRow))
				val += *itBias++;
		}

		return ret;
	}

	Matrix Linear::backward(Matrix const& dOut)
	{
		// dW = Xᵀ @ dOut
		// db = sum over batch (rows)
		// dX = dOut @ Wᵀ

		auto grads = Matrix::d_matmul(m_LastInput, m_Ws, dOut);
		m_dWs += grads.second;
		// bias

		for (size_t r = 0; r < dOut.rows(); ++r)
			for (size_t c = 0; c < dOut.cols(); ++c)
				m_dBias[c] += dOut.at(r, c);

		return grads.first;
	}

	void Linear::update(scalar lr)
	{
		for (size_t r = 0; r < m_Ws.rows(); ++r)
		{
			auto dst{ m_Ws.row(r) };
			std::transform(dst.begin(), dst.end(), m_dWs.row(r).begin(), dst.begin(), [lr](scalar a, scalar b)->scalar {return a - lr * b; });
		}

		std::transform(m_Bias.begin(), m_Bias.end(), m_dBias.begin(), m_Bias.begin(), [lr](scalar a, scalar b)->scalar {return a - lr * b; });
	}

	void Linear::zero_grad()
	{
		m_dWs.fill({});
		m_dBias.fill({});
	}
}