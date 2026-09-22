#pragma once
#include <initializer_list>
#include <vector>
#include "classmap.h"

namespace llm
{
	class Tensor
	{
		std::vector<size_t> m_Dim;
		std::vector<scalar> m_Data;

	public:
		Tensor(std::initializer_list<size_t> const&);

		//Tensor x{ 2, 3 }
		size_t ndim()const;	//   → 2
		size_t size()const;		//   → 6
		std::vector<size_t> const& shape()const;	//  →[2, 3]
		template<scalar...>
		scalar at(...)const;		// → 0
		template<scalar...>
		scalar& at(...);		// → 0
	};
}