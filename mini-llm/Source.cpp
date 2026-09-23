#include <cassert>
#include <iostream>
#include "..\mini_llm_core\core.h"

using namespace llm;

int main()
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

	//Tensor A{ 2, 3 };

	//A.set({ 1,2,3,4,5,6 });

	//Tensor B{ 3, 2 };
	//B.set({ 7,8,9,10,11,12 });

	//auto C{ A.matmul(B) };	// shape[2, 2]

	//std::cout << C;
		//58   64
		//139  154

	//assert(C.ndim() == 2);
	//assert(C.shape()[0] == 2);
	//assert(C.shape()[1] == 2);

	//assert(C.at({ 0, 0 }) == 58.0f);
	//assert(C.at({ 0, 1 }) == 64.0f);
	//assert(C.at({ 1, 0 }) == 139.0f);
	//assert(C.at({ 1, 1 }) == 154.0f);

	//Tensor left{ 2, 2 };
	//left.set({ 1, 2, 3, 4 });

	//Tensor right{ 2, 2 };
	//right.set({ 10, 20, 30, 40 });

	//auto sum = left + right;

	//assert(sum.at({ 0, 0 }) == 11.0f);
	//assert(sum.at({ 1, 1 }) == 44.0f);

	//Tensor bias{ 2 };
	//bias.set({ 10, 20 });

	//Linear l1{ B, bias };
	//C = l1.forward(A);

	//Tensor input{ 3 };
	//input.set({ -1, 0, 1 });
	//auto output{ input.gelu() };
	////: approximately[-0.1588, 0, 0.8412]

	//// Linear output expected:
	//assert(C.at({ 0, 0 }) == 68.0f);
	//assert(C.at({ 0, 1 }) == 84.0f);
	//assert(C.at({ 1, 0 }) == 149.0f);
	//assert(C.at({ 1, 1 }) == 174.0f);

	//// GELU needs a tolerance:
	//assert(std::abs(output[0] - -0.1588f) < 0.0001f);
	//assert(std::abs(output[1] - 0.0000f) < 0.0001f);
	//assert(std::abs(output[2] - 0.8412f) < 0.0001f);

	{
		Tensor identity2{ 2, 2 };
		identity2.set({ 1, 0, 0, 1 });

		Tensor zeroBias{ 2 };
		zeroBias.set({ 0, 0 });

		Linear wq{ identity2, zeroBias };
		Linear wk{ identity2, zeroBias };
		Linear wv{ identity2, zeroBias };
		Linear wo{ identity2, zeroBias };

		Attention attn{ wq, wk, wv, wo };

		Tensor attnInput{ 2, 2 };
		attnInput.set({ 1, 0, 0, 1 });

		auto attnOut{ attn.forward(attnInput) };

		assert(std::abs(attnOut.at({ 0, 0 }) - 0.6697615f) < 0.0001f);
		assert(std::abs(attnOut.at({ 0, 1 }) - 0.3302385f) < 0.0001f);
		assert(std::abs(attnOut.at({ 1, 0 }) - 0.3302385f) < 0.0001f);
		assert(std::abs(attnOut.at({ 1, 1 }) - 0.6697615f) < 0.0001f);
	}
	{
		Tensor X{ 2, 3 };
		X.set({ 1.0f, 2.0f, 0.5f,
				0.5f, -1.0f, 2.0f });

		Tensor Wq{ 3, 2 };
		Wq.set({ 0.1f, 0.2f, 0.3f, -0.1f, -0.2f, 0.4f });
		Tensor bq{ 2 };
		bq.set({ 0.05f, -0.05f });

		Tensor Wk{ 3, 2 };
		Wk.set({ 0.4f, -0.3f, 0.1f, 0.2f, 0.3f, 0.1f });
		Tensor bk{ 2 };
		bk.set({ 0.0f, 0.1f });

		Tensor Wv{ 3, 2 };
		Wv.set({ 0.2f, 0.1f, -0.1f, 0.3f, 0.5f, -0.2f });
		Tensor bv{ 2 };
		bv.set({ 0.1f, 0.0f });

		Tensor Wo{ 2, 3 };
		Wo.set({ 0.3f, -0.2f, 0.1f, 0.1f, 0.4f, -0.3f });
		Tensor bo{ 3 };
		bo.set({ 0.0f, 0.05f, -0.05f });

		Attention attn2{ Linear{ Wq, bq }, Linear{ Wk, bk }, Linear{ Wv, bv }, Linear{ Wo, bo } };

		auto result{ attn2.forward(X) };

		assert(std::abs(result.at({ 0, 0 }) - 0.24280854f) < 0.0001f);
		assert(std::abs(result.at({ 0, 1 }) - -0.11554923f) < 0.0001f);
		assert(std::abs(result.at({ 0, 2 }) - 0.03356252f) < 0.0001f);
		assert(std::abs(result.at({ 1, 0 }) - 0.23780701f) < 0.0001f);
		assert(std::abs(result.at({ 1, 1 }) - -0.09398013f) < 0.0001f);
		assert(std::abs(result.at({ 1, 2 }) - 0.01887052f) < 0.0001f);
	}
}