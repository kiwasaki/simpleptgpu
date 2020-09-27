#pragma once

#ifndef _INTERSECTION_HPP_
#define _INTERSECTION_HPP_

#include<cuda.h>
#include<helper_cuda.h>
#include<helper_math.h>

struct intersection
{
	float3	m_p;
	float3	m_n;
	float4	m_c;
};

#endif