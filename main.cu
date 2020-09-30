#include<iostream>
#include<vector>
#include<memory>
#include<chrono>

#include<curand.h>
#include<curand_kernel.h>

#include"ray.hpp"
#include"sphere.hpp"
#include"scene.hpp"
#include"image.hpp"

constexpr float pi = 3.14159265f;
constexpr float two_pi = 2.f * pi;
constexpr float inv_pi = 0.318309886f;

__global__ void make_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		int k = 0;
		//device_spheres[ 0 ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 0.1f ) );
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, -1e3f, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.2f ) ); //floor
		device_spheres[ k++ ] = new sphere( make_float3( 1e3f + 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.8f, 0.2f, 0.1f, 0.8f ) ); //right
		device_spheres[ k++ ] = new sphere( make_float3( - 1e3f - 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.1f, 0.2f, 0.8f, 0.8f ) ); //left
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 0.f, 1e3f + 15.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.1f ) ); //far
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 1e3f + 8, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.3f ) ); //ceil
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 7.0f, 10.f ), 1.f, make_float4( 5.f, 5.f, 5.f, - 1.f ) ); //light source
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 0.01f ) );
		*device_scene = new scene( device_spheres, k );
	}
}


/*
__global__ void make_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		int k = 0;
		for( int i = 0; i < 5; ++i ) {
			for( int j = 0; j < 5; ++j ) {
				const float x = ( i - 2.5f );
				const float y = ( j - 2.5f ) * 0.5f + 5.f;
				device_spheres[ k++ ] = new sphere( make_float3( x, 0.6f, y ), 0.5f, make_float4( 1.f, 1.f, 1.f, 0.1f ) );
			}
		}

		//device_spheres[ 0 ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 0.1f ) );
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, -1e3f, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.1f ) ); //floor
		device_spheres[ k++ ] = new sphere( make_float3( 1e3f + 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.8f, 0.2f, 0.1f, 0.1f ) ); //right
		device_spheres[ k++ ] = new sphere( make_float3( - 1e3f - 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.1f, 0.2f, 0.8f, 0.1f ) ); //left
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 0.f, 1e3f + 15.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.1f ) ); //far
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 1e3f + 8, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.01f ) ); //ceil
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 7.f, 10.f ), 1.f, make_float4( 10.f, 10.f, 10.f, - 1.f ) ); //light source


		*device_scene = new scene( device_spheres, k );
	}
}
*/

//
__global__ void free_scene( sphere **device_spheres, scene **device_scene )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		for( int i = 0, n = ( *device_scene )->size(); i < n; ++i ) {
			delete device_spheres[ i ];
		}
		delete *device_scene;
	}
}

//
__global__ void trace( float *pixels, scene **scene, const int width, const int height, const float3 eye )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;

	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );
	const float3 d = { m_p * ( x - width / 2.f ), m_p * ( y - height / 2.f ), 1.f };
	const ray r = { eye, normalize( d ) };

	intersection isect;

	bool hit = ( *scene )->intersect( r, isect );
	if( hit ) {
		pixels[ 3 * ( y * width + x ) + 0 ] = 0.5f * ( isect.m_n.x + 1.f );
		pixels[ 3 * ( y * width + x ) + 1 ] = 0.5f * ( isect.m_n.y + 1.f );
		pixels[ 3 * ( y * width + x ) + 2 ] = 0.5f * ( isect.m_c.z + 1.f );
	} else {
		pixels[ 3 * ( y * width + x ) + 0 ] = 0.f;
		pixels[ 3 * ( y * width + x ) + 1 ] = 0.f;
		pixels[ 3 * ( y * width + x ) + 2 ] = 0.f;
	}
}



//
__global__ void render_aa( float *pixels, scene **scene, curandState *rand_state, const int width, const int height, const float3 eye, const int ns )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;

	intersection isect;
	float3 d, L;
	ray r;
	curandState rng = rand_state[ y * width + x ];

	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );

	L = make_float3( 0.f, 0.f, 0.f );
	for( int i = 0; i < ns; ++i ) {
		d = { m_p * ( x - width / 2.f + curand_uniform( &rng ) ), m_p * ( y - height / 2.f + curand_uniform( &rng ) ), 1.f };
		r = { eye, normalize( d ) };
		if( ( *scene )->intersect( r, isect ) ) {
			L.x += 0.5f * ( isect.m_n.x + 1.f );
			L.y += 0.5f * ( isect.m_n.y + 1.f );
			L.z += 0.5f * ( isect.m_n.z + 1.f );
		}
	}
	pixels[ 3 * ( y * width + x ) + 0 ] = L.x / float( ns );
	pixels[ 3 * ( y * width + x ) + 1 ] = L.y / float( ns );
	pixels[ 3 * ( y * width + x ) + 2 ] = L.z / float( ns );
}


//
__device__ inline bool is_emissive( const intersection &isect )
{
	return ( isect.m_c.w < 0.f );
}

//
__device__ inline float3 Le( const intersection &isect )
{
	return make_float3( isect.m_c.x * inv_pi, isect.m_c.y * inv_pi, isect.m_c.z * inv_pi );
}

// sample direction for diffuse BRDF
__device__ inline float3 sample( const float3 n, const float xi1, const float xi2, float &pdf_w )
{
	const float3 t = normalize( ( std::abs( n.x ) > std::abs( n.y ) )? cross( n, make_float3( 0.f, n.z, - n.y ) ) : cross( n, make_float3( - n.z, 0.f, n.x ) ) );
	const float3 b = cross( t, n );
	const float cth = sqrt( 1.f - xi1 );
	const float sth = sqrt( max( 0.f, 1.f - cth * cth ) );
	const float cph = cos( two_pi * xi2 );
	const float sph = sin( two_pi * xi2 );
	pdf_w = inv_pi * cth;
	return cth * n + sth * cph * t + sth * sph * b;
}

//
__device__ inline float3 sample_ggx( const float3 n, const float3 wo, const float alpha, const float xi1, const float xi2, float &pdf_w )
{
	const float3 t = normalize( ( std::abs( n.x ) > std::abs( n.y ) )? cross( n, make_float3( 0.f, n.z, - n.y ) ) : cross( n, make_float3( - n.z, 0.f, n.x ) ) );
	const float3 b = cross( t, n );
	const float3 lwo = make_float3( dot( wo, t ), dot( wo, b ), dot( wo, n ) );

	//ハーフベクトルをサンプリング
	const float theta_h = atan( alpha * sqrt( xi1 / ( 1.f - xi1 ) ) );
	const float cosph = cos( two_pi * xi2 );
	const float sinph = sin( two_pi * xi2 );
	const float costh = cos( theta_h );
	const float sinth = sin( theta_h );

	const float3 h = { sinth * cosph, sinth * sinph, costh };
	const float3 lwi = 2.f * dot( lwo, h ) * h - lwo;
	const float cos2 = costh * costh;
	const float sin2 = max( 0.f, 1.f - cos2 );
	const float pdf = costh / ( pi * alpha * alpha * ( cos2 + sin2 / ( alpha * alpha ) ) * ( cos2 + sin2 / ( alpha * alpha ) ) );
	//確率密度
	pdf_w = pdf / ( 4.0 * dot( h, lwo ) );

	return lwi.x * t + lwi.y * b + lwi.z * n;
}

//
__device__ inline float3 eval_ggx( const float3 wi, const float3 wo, const float3 n, const float3 f0, const float alpha )
{
	const float3 t = normalize( ( std::abs( n.x ) > std::abs( n.y ) )? cross( n, make_float3( 0.f, n.z, - n.y ) ) : cross( n, make_float3( - n.z, 0.f, n.x ) ) );
	const float3 b = cross( t, n );
	const float3 lwo = make_float3( dot( wo, t ), dot( wo, b ), dot( wo, n ) );
	const float3 lwi = make_float3( dot( wi, t ), dot( wi, b ), dot( wi, n ) );
	float3 f = { 0.f, 0.f, 0.f };
	if( lwi.z > 0.f && lwo.z > 0.f ) {
		const float cos_i = lwi.z;
		const float cos_o = lwo.z;
		const float3 h = normalize( lwi + lwo );

		//fresnel term
		const float hh = dot( h, lwi );
		const float tmp = ( 1.f - hh ) * ( 1.f - hh ) * ( 1.f - hh ) * ( 1.f - hh ) * ( 1.f - hh );
		const float3 F = f0 + tmp * make_float3( 1.f - f0.x, 1.f - f0.y, 1.f - f0.z );

		//shadowing-masking term
		const float tani = 1.f / ( lwi.z * lwi.z ) - 1.f;
		const float tano = 1.f / ( lwo.z * lwo.z ) - 1.f;
		const float lambdai = ( - 1.f + sqrt( 1.f + alpha * alpha * tani ) ) / 2.f;
		const float lambdao = ( - 1.f + sqrt( 1.f + alpha * alpha * tano ) ) / 2.f;
		const float G = 1.f / ( 1.f + lambdai + lambdao );

		//normal distribution
		const float cosh2 = h.z * h.z;
		const float sinh2 = max( 0.f, 1.f - cosh2 );
		const float D = 1.f / ( pi * alpha * alpha * ( cosh2 + sinh2 / ( alpha * alpha ) ) * ( cosh2 + sinh2 / ( alpha * alpha ) ) );
		f = F * D * G / ( 4.f * cos_i * cos_o );
	}
	return f;
}



//
__global__ void render( float *pixels, scene **scene, curandState *rand_state, const int width, const int height, const float3 eye, const int ns )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;

	float3 L;
	float pdf_w;
	curandState rng = rand_state[ y * width + x ];
	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );
	L = make_float3( 0.f, 0.f, 0.f );
	intersection isect;
	ray r;
	float3 d;
	for( int i = 0; i < ns; ++i ) {
		d = { m_p * ( x - width / 2.f + curand_uniform( &rng ) ), m_p * ( y - height / 2.f + curand_uniform( &rng ) ), 1.f };
		r = { eye, normalize( d ) };
		float3 tp = make_float3( 1.f, 1.f, 1.f );

		if( ( *scene )->intersect( r, isect ) ) {
			//
			if( is_emissive( isect ) ) {
				L.x += isect.m_c.x;
				L.y += isect.m_c.y;
				L.z += isect.m_c.z;
			} else {
				for( int j = 0; j < 10; ++j ) {

					//d = sample( isect.m_n, curand_uniform( &rng ), curand_uniform( &rng ), pdf_w );
					//tp *= make_float3( isect.m_c.x * inv_pi, isect.m_c.y * inv_pi, isect.m_c.z * inv_pi ) * dot( d, isect.m_n ) / pdf_w;

					float3 wo = - 1.f * d;
					d = sample_ggx( isect.m_n, wo, isect.m_c.w, curand_uniform( &rng ), curand_uniform( &rng ), pdf_w );
					tp *= eval_ggx( d, wo, isect.m_n, { isect.m_c.x, isect.m_c.y, isect.m_c.z }, isect.m_c.w ) * dot( d, isect.m_n ) / pdf_w;

					r = { isect.m_p, d };
					bool hit = ( *scene )->intersect( r, isect );
					if( hit ) {
						if( is_emissive( isect ) ) {
							L += tp * Le( isect );
							break;
						}
					} else {
						break;
					}
				}
			}
		}
	}
	pixels[ 3 * ( y * width + x ) + 0 ] = L.x / float( ns );
	pixels[ 3 * ( y * width + x ) + 1 ] = L.y / float( ns );
	pixels[ 3 * ( y * width + x ) + 2 ] = L.z / float( ns );
}


//
__global__ void init( curandState *rand_state, const int width, const int height )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) || ( y >= height ) ) return;
	curand_init( y * width + x, 0, 0, &rand_state[ y * width + x ] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
	constexpr int width		= 1024;
	constexpr int height	= 1024;

	constexpr int n_object	= 7;

	float *device_buffer;

	sphere **device_spheres;
	scene  **device_scene;
	curandState *device_rand_state;

	//
	const float3 eye = { 0.f, 3.f, - 5.f };

	//
	checkCudaErrors( cudaMallocManaged( ( void ** ) &device_buffer, sizeof( float ) * 3 * width * height ) );

	//
	checkCudaErrors( cudaMalloc( ( void** ) &device_rand_state, sizeof( curandState ) * width * height ) );

	//シーンの作成
	{
		checkCudaErrors( cudaMalloc( ( void ** ) &device_spheres, n_object * sizeof( sphere * ) ) );
		checkCudaErrors( cudaMalloc( ( void ** ) &device_scene  , 1 * sizeof( scene *  ) ) );
		make_scene<<< 1, 1 >>>( device_spheres, device_scene, n_object );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	const auto start = std::chrono::system_clock::now();

	{
		dim3 grid, block;
		block.x = 16;
		block.y = 16;
		grid.x = width  / block.x;
		grid.y = height / block.y;
		checkCudaErrors( cudaMalloc( ( void ** ) &device_rand_state, width * height * sizeof( curandState ) ) );
		init<<< grid, block >>>( device_rand_state, width, height );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}


	{
		dim3 grid, block;
		block.x = 8;
		block.y = 8;
		grid.x = width  / block.x;
		grid.y = height / block.y;

		render<<< grid, block >>>( device_buffer, device_scene, device_rand_state, width, height, eye, 4096 );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	const auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast< std::chrono::milliseconds >( end - start ).count() << "ms.\n";

	{
		free_scene<<< 1, 1 >>>( device_spheres, device_scene );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	//checkCudaErrors( cudaMemcpy( host_image.get(), device_image, sizeof( float ) * 3 * width * height, cudaMemcpyDeviceToHost ) );

	save_bmp( device_buffer, width, height, "test.bmp" );

	//delete

	checkCudaErrors( cudaFree( device_rand_state ) );
	checkCudaErrors( cudaFree( device_spheres ) );
	checkCudaErrors( cudaFree( device_scene ) );
	checkCudaErrors( cudaFree( device_buffer ) );

	return 0;

}
