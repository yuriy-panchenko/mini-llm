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
		size_t ndim()const { return m_Dim.size(); }	//   → 2
		size_t size()const { return m_Data.size(); }		//   → 6
		std::vector<size_t> const& shape()const { return m_Dim; }	//  →[2, 3]

		//template<scalar...>
		//scalar at(...)const;		// → 0
		//template<scalar...>
		//scalar& at(...);		// → 0
		scalar at(std::initializer_list<size_t> adr)const { return m_Data[to_index(adr)]; }	// → 0
		scalar& at(std::initializer_list<size_t> adr) { return m_Data[to_index(adr)]; }		// → 0

	private:
		size_t to_index(std::initializer_list<size_t> const& adr)const;
	};
}