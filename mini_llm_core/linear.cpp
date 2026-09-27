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
		, m_mWs{ w.rows(), w.cols() }
		, m_vWs{ w.rows(), w.cols() }
		, m_mBias{ bias.size() }
		, m_vBias{ bias.size() }
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
		constexpr scalar beta1{ 0.9f }, beta2{ 0.999f }, eps{ 1e-8f };
		++m_T;
		auto const bc1{ 1 - std::pow(beta1, static_cast<scalar>(m_T)) };
		auto const bc2{ 1 - std::pow(beta2, static_cast<scalar>(m_T)) };

		for (size_t r = 0; r < m_Ws.rows(); ++r)
		{
			auto w{ m_Ws.row(r) }, g{ m_dWs.row(r) }, m{ m_mWs.row(r) }, v{ m_vWs.row(r) };

			for (size_t c = 0; c < m_Ws.cols(); ++c)
			{
				m[c] = beta1 * m[c] + (1 - beta1) * g[c];
				v[c] = beta2 * v[c] + (1 - beta2) * g[c] * g[c];

				auto const mHat{ m[c] / bc1 };
				auto const vHat{ v[c] / bc2 };

				w[c] -= lr * mHat / (std::sqrt(vHat) + eps);
			}
		}

		for (size_t c = 0; c < m_Bias.size(); ++c)
		{
			m_mBias[c] = beta1 * m_mBias[c] + (1 - beta1) * m_dBias[c];
			m_vBias[c] = beta2 * m_vBias[c] + (1 - beta2) * m_dBias[c] * m_dBias[c];

			auto const mHat{ m_mBias[c] / bc1 };
			auto const vHat{ m_vBias[c] / bc2 };

			m_Bias[c] -= lr * mHat / (std::sqrt(vHat) + eps);
		}
	}

	void Linear::zero_grad()
	{
		m_dWs.fill({});
		m_dBias.fill({});
		// m_mWs/m_vWs/m_mBias/m_vBias intentionally NOT cleared here --
		// Adam's moment estimates are meant to persist across steps.
	}
}