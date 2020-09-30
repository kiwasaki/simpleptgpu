#pragma once

#ifndef _SCENE_HPP_
#define _SCENE_HPP_

#include<cuda.h>
#include<helper_cuda.h>
#include<helper_math.h>

#include"sphere.hpp"
#include"ray.hpp"
#include"intersection.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class scene : public sphere
{
public:

	__device__ scene() {}

	__device__ scene( sphere **obj, const int n ) : m_n( n ) { m_object = obj; }

	__device__ bool intersect( const ray &r, intersection &isect ) const
	{
		float mint = 1e10f;
		bool hit = false;
		int id = 0;

		for( int i = 0; i < m_n; ++i ) {
			float t = m_object[ i ]->intersect( r );
			if( t < mint && t > 0.f ) {
				mint = t;
				hit = true;
				id = i;
			}
		}

		if( hit ) {
			isect.m_p = r.o() + mint * r.d();
			isect.m_n = normalize( isect.m_p - m_object[ id ]->center() );
			isect.m_c = m_object[ id ]->color();
		}
		return hit;
	}

	__device__ int size() const { return m_n; }

private:
	sphere		**m_object;
	int 		m_n;
};



#endif