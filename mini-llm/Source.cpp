#include <cassert>
#include <iostream>
#include "..\mini_llm_core\core.h"

using namespace llm;

int main()
{
	{
		//Tensor u{}, u2{ 4, 0, 3 };
		//Tensor u{ 2, 3 }, u2{ 4, 2, 3 };

		//Tensor x{ 2, 3 };

		//auto r1 = x.ndim();//   → 2
		//auto r2 = x.size();//   → 6
		//auto r3 = x.shape();//  →[2, 3]
		//auto r4 = x.at({ 0, 0 });// → 0
		//auto r5 = x.at({ 1, 2 });// → 0

		//Tensor{ 2, 3 }.at({ 1, 2 });//  valid → flat index 5
		//Tensor{ 2, 3 }.at({ 2, 0 });//  invalid: row 2 does not exist
		//Tensor{ 2, 3 }.at({ 1 });//     invalid: wrong number of indices
		//Tensor{ 2, 3 }.at({ 1, 2, 0 });// invalid: wrong number of indices
		//x.at({ 1, 2 }) = 42.0f;
		//x[5] = 42.0f;

		//x.at({ 1, 2 }) = 42.0f;
		//assert(x[5] == 42.0f);

		//x[0] = 7.0f;
		//assert(x.at({ 0, 0 }) == 7.0f);

		Matrix A{ 2, 3 };

		A.set({ 1,2,3,4,5,6 });

		Matrix B{ 3, 2 };
		B.set({ 7,8,9,10,11,12 });

		auto C{ A.matmul(B) };	// shape[2, 2]

		//std::cout << C;
			//58   64
			//139  154

		//assert(C.ndim() == 2);
		assert(C.rows() == 2);
		assert(C.cols() == 2);

		assert(C.at(0, 0) == 58.0f);
		assert(C.at(0, 1) == 64.0f);
		assert(C.at(1, 0) == 139.0f);
		assert(C.at(1, 1) == 154.0f);

		Matrix left{ 2, 2 };
		left.set({ 1, 2, 3, 4 });

		Matrix right{ 2, 2 };
		right.set({ 10, 20, 30, 40 });

		auto sum = left + right;

		assert(sum.at(0, 0) == 11.0f);
		assert(sum.at(1, 1) == 44.0f);

		Vector bias{ 2 };
		bias.set({ 10, 20 });

		Linear l1{ B, bias };
		C = l1.forward(A);

		Vector input{ 3 };
		input.set({ -1, 0, 1 });
		auto output{ input.gelu() };
		//: approximately[-0.1588, 0, 0.8412]

		// Linear output expected:
		assert(C.at(0, 0) == 68.0f);
		assert(C.at(0, 1) == 84.0f);
		assert(C.at(1, 0) == 149.0f);
		assert(C.at(1, 1) == 174.0f);

		// GELU needs a tolerance:
		assert(std::abs(output[0] - -0.1588f) < 0.0001f);
		assert(std::abs(output[1] - 0.0000f) < 0.0001f);
		assert(std::abs(output[2] - 0.8412f) < 0.0001f);
	}

	{
		Matrix identity2{ 2, 2 };
		identity2.set({ 1, 0, 0, 1 });

		Vector zeroBias{ 2 };
		zeroBias.set({ 0, 0 });

		Linear wq{ identity2, zeroBias };
		Linear wk{ identity2, zeroBias };
		Linear wv{ identity2, zeroBias };
		Linear wo{ identity2, zeroBias };

		Attention attn{ wq, wk, wv, wo };

		Matrix attnInput{ 2, 2 };
		attnInput.set({ 1, 0, 0, 1 });

		auto attnOut{ attn.forward(attnInput) };

		assert(std::abs(attnOut.at(0, 0) - 0.6697615f) < 0.0001f);
		assert(std::abs(attnOut.at(0, 1) - 0.3302385f) < 0.0001f);
		assert(std::abs(attnOut.at(1, 0) - 0.3302385f) < 0.0001f);
		assert(std::abs(attnOut.at(1, 1) - 0.6697615f) < 0.0001f);
	}
	{
		Matrix X{ 2, 3 };
		X.set({ 1.0f, 2.0f, 0.5f,
				0.5f, -1.0f, 2.0f });

		Matrix Wq{ 3, 2 };
		Wq.set({ 0.1f, 0.2f, 0.3f, -0.1f, -0.2f, 0.4f });
		Vector bq{ 2 };
		bq.set({ 0.05f, -0.05f });

		Matrix Wk{ 3, 2 };
		Wk.set({ 0.4f, -0.3f, 0.1f, 0.2f, 0.3f, 0.1f });
		Vector bk{ 2 };
		bk.set({ 0.0f, 0.1f });

		Matrix Wv{ 3, 2 };
		Wv.set({ 0.2f, 0.1f, -0.1f, 0.3f, 0.5f, -0.2f });
		Vector bv{ 2 };
		bv.set({ 0.1f, 0.0f });

		Matrix Wo{ 2, 3 };
		Wo.set({ 0.3f, -0.2f, 0.1f, 0.1f, 0.4f, -0.3f });
		Vector bo{ 3 };
		bo.set({ 0.0f, 0.05f, -0.05f });

		Attention attn2{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo } };

		auto result{ attn2.forward(X) };

		assert(std::abs(result.at(0, 0) - 0.24280854f) < 0.0001f);
		assert(std::abs(result.at(0, 1) - -0.11554923f) < 0.0001f);
		assert(std::abs(result.at(0, 2) - 0.03356252f) < 0.0001f);
		assert(std::abs(result.at(1, 0) - 0.23780701f) < 0.0001f);
		assert(std::abs(result.at(1, 1) - -0.09398013f) < 0.0001f);
		assert(std::abs(result.at(1, 2) - 0.01887052f) < 0.0001f);
	}
	{
		Matrix A{ 2, 3 };
		A.set({ 1, 2, 3, 4, 5, 6 });

		auto T{ A.transpose() };

		assert(T.rows() == 3);
		assert(T.cols() == 2);

		assert(T.at(0, 0) == 1.0f);
		assert(T.at(0, 1) == 4.0f);
		assert(T.at(1, 0) == 2.0f);
		assert(T.at(1, 1) == 5.0f);
		assert(T.at(2, 0) == 3.0f);
		assert(T.at(2, 1) == 6.0f);
	}
	{
		Matrix X{ 2, 4 };
		X.set({ 1.00000000f, 0.50000000f, -1.00000000f, 2.00000000f,
				0.20000000f, -0.30000001f, 0.69999999f, 0.10000000f });

		Matrix Wq{ 4, 4 };
		Wq.set({ 0.10000000f, 0.20000000f, -0.10000000f, 0.05000000f,
				 0.30000001f, -0.20000000f, 0.10000000f, 0.00000000f,
				 -0.10000000f, 0.40000001f, 0.20000000f, -0.30000001f,
				 0.05000000f, 0.10000000f, -0.20000000f, 0.30000001f });
		Vector bq{ 4 };
		bq.set({ 0.10000000f, -0.10000000f, 0.05000000f, 0.00000000f });

		Matrix Wk{ 4, 4 };
		Wk.set({ 0.20000000f, -0.10000000f, 0.30000001f, 0.10000000f,
				 0.10000000f, 0.30000001f, -0.20000000f, 0.05000000f,
				 -0.30000001f, 0.20000000f, 0.10000000f, 0.40000001f,
				 0.40000001f, -0.05000000f, 0.20000000f, -0.10000000f });
		Vector bk{ 4 };
		bk.set({ 0.00000000f, 0.05000000f, -0.05000000f, 0.10000000f });

		Matrix Wv{ 4, 4 };
		Wv.set({ 0.15000001f, 0.25000000f, -0.10000000f, 0.20000000f,
				 -0.20000000f, 0.10000000f, 0.30000001f, -0.15000001f,
				 0.10000000f, -0.30000001f, 0.20000000f, 0.05000000f,
				 0.05000000f, 0.20000000f, -0.25000000f, 0.10000000f });
		Vector bv{ 4 };
		bv.set({ 0.05000000f, 0.00000000f, 0.10000000f, -0.05000000f });

		Matrix Wo{ 4, 4 };
		Wo.set({ 0.20000000f, -0.10000000f, 0.30000001f, 0.05000000f,
				 0.10000000f, 0.30000001f, -0.20000000f, 0.15000001f,
				 -0.15000001f, 0.20000000f, 0.10000000f, -0.10000000f,
				 0.05000000f, -0.05000000f, 0.20000000f, 0.25000000f });
		Vector bo{ 4 };
		bo.set({ 0.00000000f, 0.10000000f, -0.05000000f, 0.05000000f });

		MultiHeadAttention mha{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo }, 2 };

		auto result{ mha.forward(X) };

		assert(std::abs(result.at(0, 0) - 0.11223950f) < 0.0001f);
		assert(std::abs(result.at(0, 1) - 0.23256811f) < 0.0001f);
		assert(std::abs(result.at(0, 2) - -0.11266868f) < 0.0001f);
		assert(std::abs(result.at(0, 3) - 0.19079345f) < 0.0001f);
		assert(std::abs(result.at(1, 0) - 0.11482176f) < 0.0001f);
		assert(std::abs(result.at(1, 1) - 0.14417185f) < 0.0001f);
		assert(std::abs(result.at(1, 2) - -0.07225687f) < 0.0001f);
		assert(std::abs(result.at(1, 3) - 0.17943997f) < 0.0001f);
	}
	{
		Matrix X{ 2, 3 };
		X.set({ 0.50000000f, -0.20000000f, 1.00000000f, -1.00000000f, 0.30000001f, 0.20000000f });

		Matrix Wq{ 3, 2 };
		Wq.set({ 0.10000000f, -0.20000000f, 0.30000001f, 0.10000000f, -0.10000000f, 0.20000000f });
		Vector bq{ 2 };
		bq.set({ 0.05000000f, -0.05000000f });

		Matrix Wk{ 3, 2 };
		Wk.set({ 0.20000000f, 0.10000000f, -0.10000000f, 0.30000001f, 0.15000001f, -0.20000000f });
		Vector bk{ 2 };
		bk.set({ 0.00000000f, 0.10000000f });

		Matrix Wv{ 3, 2 };
		Wv.set({ 0.10000000f, 0.25000000f, 0.20000000f, -0.10000000f, -0.15000001f, 0.05000000f });
		Vector bv{ 2 };
		bv.set({ 0.05000000f, 0.00000000f });

		Matrix Wo{ 2, 3 };
		Wo.set({ 0.20000000f, -0.10000000f, 0.15000001f, 0.10000000f, 0.30000001f, -0.05000000f });
		Vector bo{ 3 };
		bo.set({ 0.00000000f, 0.05000000f, -0.05000000f });

		Vector gamma1{ 3 };
		gamma1.set({ 1.00000000f, 1.00000000f, 1.00000000f });

		Matrix Wf1{ 3, 4 };
		Wf1.set({ 0.10000000f, -0.20000000f, 0.15000001f, 0.05000000f,
				  0.20000000f, 0.10000000f, -0.10000000f, 0.30000001f,
				  -0.15000001f, 0.25000000f, 0.20000000f, -0.05000000f });
		Vector bf1{ 4 };
		bf1.set({ 0.05000000f, -0.05000000f, 0.10000000f, 0.00000000f });

		Matrix Wf2{ 4, 3 };
		Wf2.set({ 0.20000000f, -0.10000000f, 0.15000001f,
				  0.10000000f, 0.30000001f, -0.05000000f,
				  -0.20000000f, 0.05000000f, 0.25000000f,
				  0.15000001f, -0.15000001f, 0.10000000f });
		Vector bf2{ 3 };
		bf2.set({ 0.00000000f, 0.05000000f, -0.05000000f });

		Vector gamma2{ 3 };
		gamma2.set({ 1.00000000f, 1.00000000f, 1.00000000f });

		TransformerBlock<Attention> block{
			Attention{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo } },
			RMSNorm{ gamma1 },
			Linear{ Wf1, bf1 },
			Linear{ Wf2, bf2 },
			RMSNorm{ gamma2 }
		};

		auto result{ block.forward(X) };

		assert(std::abs(result.at(0, 0) - 0.37835372f) < 0.0001f);
		assert(std::abs(result.at(0, 1) - -0.05428675f) < 0.0001f);
		assert(std::abs(result.at(0, 2) - 0.96446670f) < 0.0001f);
		assert(std::abs(result.at(1, 0) - -0.99071364f) < 0.0001f);
		assert(std::abs(result.at(1, 1) - 0.45434434f) < 0.0001f);
		assert(std::abs(result.at(1, 2) - 0.05856703f) < 0.0001f);
	}
	{
		Matrix table{ 4, 3 };
		table.set({ 1.0f, 2.0f, 3.0f,      // row 0
					10.0f, 20.0f, 30.0f,   // row 1
					100.0f, 200.0f, 300.0f,// row 2
					-1.0f, -2.0f, -3.0f }); // row 3

		Embedding emb{ table };

		auto result{ emb.forward({ 2, 0, 3 }) };

		assert(result.rows() == 3);
		assert(result.cols() == 3);

		assert(result.at(0, 0) == 100.0f);
		assert(result.at(0, 1) == 200.0f);
		assert(result.at(0, 2) == 300.0f);

		assert(result.at(1, 0) == 1.0f);
		assert(result.at(1, 1) == 2.0f);
		assert(result.at(1, 2) == 3.0f);

		assert(result.at(2, 0) == -1.0f);
		assert(result.at(2, 1) == -2.0f);
		assert(result.at(2, 2) == -3.0f);
	}
	{
		// Matrix::matmul, isolated
		Matrix A{ 2, 3 };
		A.set({ 1, 2, 3, 4, 5, 6 });

		Matrix B{ 3, 2 };
		B.set({ 7, 8, 9, 10, 11, 12 });

		auto C{ A.matmul(B) };

		assert(C.rows() == 2);
		assert(C.cols() == 2);
		assert(std::abs(C.at(0, 0) - 58.0f) < 0.0001f);
		assert(std::abs(C.at(0, 1) - 64.0f) < 0.0001f);
		assert(std::abs(C.at(1, 0) - 139.0f) < 0.0001f);
		assert(std::abs(C.at(1, 1) - 154.0f) < 0.0001f);
	}
	{
		// Matrix::gelu, isolated
		Matrix X{ 1, 7 };
		X.set({ -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f });

		auto Y{ X.gelu() };

		assert(std::abs(Y.at(0, 0) - -0.0454023f) < 0.0001f);
		assert(std::abs(Y.at(0, 1) - -0.1588080f) < 0.0001f);
		assert(std::abs(Y.at(0, 2) - -0.1542860f) < 0.0001f);
		assert(std::abs(Y.at(0, 3) - 0.0000000f) < 0.0001f);
		assert(std::abs(Y.at(0, 4) - 0.3457140f) < 0.0001f);
		assert(std::abs(Y.at(0, 5) - 0.8411920f) < 0.0001f);
		assert(std::abs(Y.at(0, 6) - 1.9545977f) < 0.0001f);
	}
	{
		// Matrix::softmax, isolated
		Matrix S{ 2, 3 };
		S.set({ 1.0f, 2.0f, 3.0f,
				1.0f, 1.0f, 1.0f });

		auto P{ S.softmax() };

		assert(std::abs(P.at(0, 0) - 0.0900306f) < 0.0001f);
		assert(std::abs(P.at(0, 1) - 0.2447285f) < 0.0001f);
		assert(std::abs(P.at(0, 2) - 0.6652409f) < 0.0001f);
		assert(std::abs(P.at(1, 0) - (1.0f / 3.0f)) < 0.0001f);
		assert(std::abs(P.at(1, 1) - (1.0f / 3.0f)) < 0.0001f);
		assert(std::abs(P.at(1, 2) - (1.0f / 3.0f)) < 0.0001f);
	}
	{
		// Matrix::slice_cols / Matrix::concat_cols, isolated
		Matrix M{ 2, 4 };
		M.set({ 1, 2, 3, 4,
				5, 6, 7, 8 });

		auto left{ M.slice_cols(0, 2) };
		auto right{ M.slice_cols(2, 2) };

		assert(left.rows() == 2 && left.cols() == 2);
		assert(std::abs(left.at(0, 0) - 1.0f) < 0.0001f);
		assert(std::abs(left.at(0, 1) - 2.0f) < 0.0001f);
		assert(std::abs(left.at(1, 0) - 5.0f) < 0.0001f);
		assert(std::abs(left.at(1, 1) - 6.0f) < 0.0001f);

		assert(right.rows() == 2 && right.cols() == 2);
		assert(std::abs(right.at(0, 0) - 3.0f) < 0.0001f);
		assert(std::abs(right.at(0, 1) - 4.0f) < 0.0001f);
		assert(std::abs(right.at(1, 0) - 7.0f) < 0.0001f);
		assert(std::abs(right.at(1, 1) - 8.0f) < 0.0001f);

		auto rejoined{ Matrix::concat_cols({ left, right }) };

		assert(rejoined.rows() == M.rows());
		assert(rejoined.cols() == M.cols());
		for (size_t r = 0; r < M.rows(); ++r)
			for (size_t c = 0; c < M.cols(); ++c)
				assert(std::abs(rejoined.at(r, c) - M.at(r, c)) < 0.0001f);
	}
	{
		// Attention with causal masking — same weights as the non-identity Attention test above,
		// but row 0 is now restricted to attending only to itself.
		Matrix X{ 2, 3 };
		X.set({ 1.0f, 2.0f, 0.5f,
				0.5f, -1.0f, 2.0f });

		Matrix Wq{ 3, 2 };
		Wq.set({ 0.1f, 0.2f, 0.3f, -0.1f, -0.2f, 0.4f });
		Vector bq{ 2 };
		bq.set({ 0.05f, -0.05f });

		Matrix Wk{ 3, 2 };
		Wk.set({ 0.4f, -0.3f, 0.1f, 0.2f, 0.3f, 0.1f });
		Vector bk{ 2 };
		bk.set({ 0.0f, 0.1f });

		Matrix Wv{ 3, 2 };
		Wv.set({ 0.2f, 0.1f, -0.1f, 0.3f, 0.5f, -0.2f });
		Vector bv{ 2 };
		bv.set({ 0.1f, 0.0f });

		Matrix Wo{ 2, 3 };
		Wo.set({ 0.3f, -0.2f, 0.1f, 0.1f, 0.4f, -0.3f });
		Vector bo{ 3 };
		bo.set({ 0.0f, 0.05f, -0.05f });

		Attention causalAttn{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo }, /*causal=*/true };

		auto result{ causalAttn.forward(X) };

		// Row 0 can only attend to itself (attn weights [1, 0]) — its output no longer
		// blends in row 1's Value at all, which is why this differs from the unmasked test.
		assert(std::abs(result.at(0, 0) - 0.16500001f) < 0.0001f);
		assert(std::abs(result.at(0, 1) - 0.22000001f) < 0.0001f);
		assert(std::abs(result.at(0, 2) - -0.19500001f) < 0.0001f);

		// Row 1 can see everything up to and including itself, i.e. the whole sequence here —
		// so it's identical to the unmasked result.
		assert(std::abs(result.at(1, 0) - 0.23780701f) < 0.0001f);
		assert(std::abs(result.at(1, 1) - -0.09398013f) < 0.0001f);
		assert(std::abs(result.at(1, 2) - 0.01887052f) < 0.0001f);
	}
	{
		// sinusoidal_positional_encoding, isolated
		auto pe{ sinusoidal_positional_encoding(3, 4) };

		assert(pe.rows() == 3);
		assert(pe.cols() == 4);

		// pos 0: sin(0)=0, cos(0)=1 in every pair
		assert(std::abs(pe.at(0, 0) - 0.0f) < 0.0001f);
		assert(std::abs(pe.at(0, 1) - 1.0f) < 0.0001f);
		assert(std::abs(pe.at(0, 2) - 0.0f) < 0.0001f);
		assert(std::abs(pe.at(0, 3) - 1.0f) < 0.0001f);

		assert(std::abs(pe.at(1, 0) - 0.8414710f) < 0.0001f);
		assert(std::abs(pe.at(1, 1) - 0.5403023f) < 0.0001f);
		assert(std::abs(pe.at(1, 2) - 0.0099998f) < 0.0001f);
		assert(std::abs(pe.at(1, 3) - 0.9999500f) < 0.0001f);

		assert(std::abs(pe.at(2, 0) - 0.9092974f) < 0.0001f);
		assert(std::abs(pe.at(2, 1) - -0.4161468f) < 0.0001f);
		assert(std::abs(pe.at(2, 2) - 0.0199987f) < 0.0001f);
		assert(std::abs(pe.at(2, 3) - 0.9998000f) < 0.0001f);
	}
	{
		// Model<Attention>: Embedding + sinusoidal positional encoding + one causal
		// TransformerBlock + output head, end to end. Reference computed via numpy in float32.
		size_t const vocabSize{ 4 }, dModel{ 4 }, dHidden{ 4 };

		Matrix table{ vocabSize, dModel };
		table.set({ -0.251f,  0.901f,  0.464f,  0.197f,
					-0.688f, -0.688f, -0.884f,  0.732f,
					 0.202f,  0.416f, -0.959f,  0.940f,
					 0.665f, -0.575f, -0.636f, -0.633f });
		Embedding tokenEmbedding{ table };

		Matrix Wq{ dModel, dModel };
		Wq.set({ -0.392f,  0.050f, -0.136f, -0.418f,
				  0.224f, -0.721f, -0.416f, -0.267f,
				 -0.088f,  0.570f, -0.601f,  0.028f,
				  0.185f, -0.907f,  0.215f, -0.659f });
		Vector bq{ dModel };
		bq.set({ -0.174f, 0.180f, 0.186f, 0.123f });

		Matrix Wk{ dModel, dModel };
		Wk.set({ -0.391f, -0.805f,  0.368f, -0.120f,
				 -0.756f, -0.010f, -0.931f,  0.819f,
				 -0.482f,  0.325f, -0.377f,  0.040f,
				  0.093f, -0.630f,  0.939f,  0.550f });
		Vector bk{ dModel };
		bk.set({ 0.176f, 0.158f, 0.039f, 0.169f });

		Matrix Wv{ dModel, dModel };
		Wv.set({ -0.823f, -0.608f, -0.910f, -0.349f,
				 -0.223f, -0.457f,  0.657f, -0.286f,
				 -0.438f,  0.085f, -0.718f,  0.604f,
				 -0.851f,  0.974f,  0.544f, -0.603f });
		Vector bv{ dModel };
		bv.set({ -0.198f, 0.126f, 0.083f, 0.092f });

		Matrix Wo{ dModel, dModel };
		Wo.set({ 0.543f, -0.852f, -0.283f, -0.768f,
				  0.726f,  0.247f, -0.338f, -0.873f,
				 -0.378f, -0.350f,  0.459f,  0.275f,
				  0.774f, -0.056f, -0.761f,  0.426f });
		Vector bo{ dModel };
		bo.set({ 0.104f, 0.025f, 0.108f, -0.002f });

		Attention causalAttn{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo }, /*causal=*/true };

		Vector gamma1{ dModel };
		gamma1.set({ 1.009f, 0.971f, 0.810f, 0.843f });
		RMSNorm norm1{ gamma1 };

		Matrix Wf1{ dModel, dHidden };
		Wf1.set({ -0.937f,  0.273f, -0.371f,  0.017f,
				   0.815f, -0.501f, -0.179f,  0.511f,
				  -0.542f, -0.846f, -0.420f, -0.678f,
				   0.859f,  0.616f,  0.267f,  0.743f });
		Vector bf1{ dHidden };
		bf1.set({ 0.121f, -0.125f, 0.157f, 0.016f });

		Matrix Wf2{ dHidden, dModel };
		Wf2.set({ 0.615f,  0.792f, -0.364f, -0.780f,
				  -0.544f, -0.146f,  0.636f,  0.721f,
				  -0.986f,  0.021f, -0.165f, -0.556f,
				  -0.760f, -0.325f,  0.886f, -0.354f });
		Vector bf2{ dModel };
		bf2.set({ 0.008f, 0.081f, -0.055f, 0.189f });

		Vector gamma2{ dModel };
		gamma2.set({ 1.185f, 0.901f, 0.999f, 0.920f });
		RMSNorm norm2{ gamma2 };

		TransformerBlock<Attention> block{ causalAttn, norm1, Linear{ Wf1, bf1 }, Linear{ Wf2, bf2 }, norm2 };

		Matrix Wout{ dModel, vocabSize };
		Wout.set({ -0.430f, -0.926f,  0.219f,  0.005f,
				   -0.897f, -0.443f,  0.817f, -0.521f,
				   -0.710f, -0.021f,  0.971f, -0.516f,
					0.344f,  0.523f, -0.525f,  0.456f });
		Vector bout{ vocabSize };
		bout.set({ -0.053f, 0.053f, 0.053f, 0.014f });

		Model<Attention> model{ tokenEmbedding, std::vector<TransformerBlock<Attention>>{ block }, Linear{ Wout, bout } };

		auto logits{ model.forward({ 2, 0, 3 }) };

		assert(logits.rows() == 3);
		assert(logits.cols() == vocabSize);

		assert(std::abs(logits.at(0, 0) - -3.9641535f) < 0.001f);
		assert(std::abs(logits.at(0, 1) - -1.0988278f) < 0.001f);
		assert(std::abs(logits.at(0, 2) - 4.0095810f) < 0.001f);
		assert(std::abs(logits.at(0, 3) - -2.5665348f) < 0.001f);

		assert(std::abs(logits.at(1, 0) - -4.6517696f) < 0.001f);
		assert(std::abs(logits.at(1, 1) - -1.1724832f) < 0.001f);
		assert(std::abs(logits.at(1, 2) - 4.7731414f) < 0.001f);
		assert(std::abs(logits.at(1, 3) - -2.7810917f) < 0.001f);

		assert(std::abs(logits.at(2, 0) - -1.4987745f) < 0.001f);
		assert(std::abs(logits.at(2, 1) - -0.5249774f) < 0.001f);
		assert(std::abs(logits.at(2, 2) - 1.5039064f) < 0.001f);
		assert(std::abs(logits.at(2, 3) - -0.7458587f) < 0.001f);
	}
}