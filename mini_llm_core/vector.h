#pragma once
#include "classmap.h"

namespace llm
{

	class Vector
		:public vdb
	{
	public:
		Vector(size_t elem)
			:vdb(elem, scalar{})
		{}

		void set(vdb const& data) { assert(data.size() == size()); ((vdb&)*this) = data; }
		void set(vdb&& data) { assert(data.size() == size()); ((vdb&)*this) = std::move(data); }

		Vector gelu()const { auto ret{ *this }; std::transform(ret.begin(), ret.end(), ret.begin(), GELU); return ret; }
	};
}