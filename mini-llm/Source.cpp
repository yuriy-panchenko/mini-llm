#include <cassert>
#include <iostream>
#include "..\mini_llm_core\core.h"

using namespace llm;

int main()
{
	//Tensor u{}, u2{ 4, 0, 3 };
	//Tensor u{ 2, 3 }, u2{ 4, 2, 3 };

	Tensor x{ 2, 3 };

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

	Tensor A{ 2, 3 };

	A.set({ 1,2,3,4,5,6 });

	Tensor B{ 3, 2 };
	B.set({ 7,8,9,10,11,12 });

	auto C{ matmul(A, B) };	// shape[2, 2]

	//std::cout << C;
		//58   64
		//139  154

	assert(C.ndim() == 2);
	assert(C.shape()[0] == 2);
	assert(C.shape()[1] == 2);

	assert(C.at({ 0, 0 }) == 58.0f);
	assert(C.at({ 0, 1 }) == 64.0f);
	assert(C.at({ 1, 0 }) == 139.0f);
	assert(C.at({ 1, 1 }) == 154.0f);

	Tensor left{ 2, 2 };
	left.set({ 1, 2, 3, 4 });

	Tensor right{ 2, 2 };
	right.set({ 10, 20, 30, 40 });

	auto sum = add(left, right);

	assert(sum.at({ 0, 0 }) == 11.0f);
	assert(sum.at({ 1, 1 }) == 44.0f);
}