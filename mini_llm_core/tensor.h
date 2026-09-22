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
	};
}