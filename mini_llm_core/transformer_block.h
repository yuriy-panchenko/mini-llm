#pragma once
#include "rmsnorm.h"
#include "layer.h"

namespace llm
{
	template<typename AttnT>
	class TransformerBlock
		:public Layer
	{
		AttnT m_Attn;
		RMSNorm m_Norm1;
		Linear m_Ffn1, m_Ffn2;
		RMSNorm m_Norm2;

		Matrix m_LastFfn1PreGelu;

	public:
		TransformerBlock(AttnT const& attn, RMSNorm const& norm1,
			Linear const& ffn1, Linear const& ffn2, RMSNorm const& norm2)
			:m_Attn{ attn }
			, m_Norm1{ norm1 }
			, m_Ffn1{ ffn1 }
			, m_Ffn2{ ffn2 }
			, m_Norm2{ norm2 }
			, m_LastFfn1PreGelu{ 0, 0 }
		{}

		Matrix forward(Matrix const& input)
		{
			auto const resid1{ input + m_Attn.forward(m_Norm1.forward(input)) };

			m_LastFfn1PreGelu = m_Ffn1.forward(m_Norm2.forward(resid1));
			auto const ffnOut{ m_Ffn2.forward(m_LastFfn1PreGelu.gelu()) };

			return resid1 + ffnOut;
		}

		Matrix backward(Matrix const& dOut)
		{
			// resid2 = resid1 + Ffn2(gelu(Ffn1(Norm2(resid1))))
			auto dResid1{ dOut };   // direct residual path: resid1 -> resid2 unchanged

			auto dGeluOut{ m_Ffn2.backward(dOut) };
			auto dPreGelu{ m_LastFfn1PreGelu.d_gelu(dGeluOut) };
			auto dNorm2In{ m_Ffn1.backward(dPreGelu) };
			dResid1 += m_Norm2.backward(dNorm2In);   // + gradient via the FFN branch

			// resid1 = input + Attn(Norm1(input))
			auto dInput{ dResid1 };   // direct residual path: input -> resid1 unchanged

			auto dNorm1Out{ m_Attn.backward(dResid1) };
			dInput += m_Norm1.backward(dNorm1Out);   // + gradient via the attention branch

			return dInput;
		}

		void update(scalar lr) override
		{
			m_Attn.update(lr);
			m_Norm1.update(lr);
			m_Ffn1.update(lr);
			m_Ffn2.update(lr);
			m_Norm2.update(lr);
		}

		void zero_grad() override
		{
			m_Attn.zero_grad();
			m_Norm1.zero_grad();
			m_Ffn1.zero_grad();
			m_Ffn2.zero_grad();
			m_Norm2.zero_grad();
		}
	};
}