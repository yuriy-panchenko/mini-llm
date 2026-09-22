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
	x.at({ 1, 2 }) = 42.0f;
	x[5] = 42.0f;
}