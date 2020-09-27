#pragma once

#ifndef _RAY_HPP_
#define _RAY_HPP_

#include<cuda.h>
#include<helper_cuda.h>
#include<helper_math.h>

//
class ray
{
public:

	__device__ ray() = default;

	__device__ ray( const float3 &o, const float3 &d ) : m_o( o ), m_d( d ) {}

	__device__ float3 o() const { return m_o; }

	__device__ float3 d() const { return m_d; }

private:
	float3	m_o;
	float3	m_d;
};



#endif