#pragma once

#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_

#include<cuda.h>
#include<helper_cuda.h>
#include<helper_math.h>

#include"ray.hpp"
#include"intersection.hpp"

constexpr float eps = 5e-4f;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class object
{
public:
	__device__ virtual float intersect( const ray &r ) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class sphere
{
public:

	__device__ sphere() : m_center(), m_radius(), m_color() {}

	__device__ sphere( const float3 &c, const float r, const float4 &col ) : m_center( c ), m_radius( r ), m_color( col ) {}

	__device__ float intersect( const ray &r ) const;

	__device__ float3 center() const { return m_center; }

	__device__ float4 color() const { return m_color; }

	__device__ float radius() const { return m_radius; }

private:
	float3	m_center;
	float 	m_radius;
	float4	m_color;
};

/*
__device__ float sphere::intersect( const ray &r ) const
{
	float3 p = m_center - r.o();
	const float b = dot( p, r.d() );
	const float det = b * b - dot( p, p ) + m_radius * m_radius;
	if( det >= 0.f ) {
		const float sqrt_det = sqrt( det );
		const float t1 = b - sqrt_det;
		const float t2 = b + sqrt_det;
		if( t1 > eps ) {
			return t1;
		} else if( t2 > eps ) {
			return t2;
		}
	}
	return 0.0f;
}
*/

__device__ float sphere::intersect( const ray &r ) const
{
	float3 p = m_center - r.o();
	const float b = dot( p, r.d() );
	const float det = b * b - dot( p, p ) + m_radius * m_radius;
	float t = 0.f;
	if( det >= 0.f ) {
		const float sqrt_det = sqrt( det );
		const float t1 = b - sqrt_det;
		const float t2 = b + sqrt_det;
		if( t1 > eps ) {
			t = t1;
		} else if( t2 > eps ) {
			t = t2;
		}
	}
	return t;
}

#endif