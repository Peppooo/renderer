#include "algebra.cuh"

enum class material_type : uint8_t {
	diffuse,
	specular
};

__device__ vec3 randomVecInHemisphere(const vec3& n,curandStatePhilox4_32_10_t* state) {
	vec3 v;
	while(true) {
		v = randomVec(state) * 2 - vec3{1,1,1};
		if(dot(v,n) > 0) return v;
	}
}

class material {
public:
	material_type type;
	bool isEmissive;
	material() {};
	material(material_type Type,bool IsEmissive = 0): type(Type),isEmissive(IsEmissive) {};
	__device__ __forceinline__ vec3 bounce(const vec3& D,vec3 n,curandStatePhilox4_32_10_t* state) const {
		if(type == material_type::specular) {
			return D-n*2*dot(D,n);
		}
		else if(type== material_type::diffuse) {
			float r1 = curand_uniform(state);
			float r2 = curand_uniform(state);

			float phi = 2.0f * M_PI * r1;
			float r = sqrtf(r2);

			float x = r * cos(phi);
			float y = r * sin(phi);
			float z = sqrtf(1.0f - r2);

			vec3 t = any_perpendicular(n);
			vec3 b = cross(t,n);
			
			return (t*x + b*y + n*z);
		}
		return vec3{0,0,0};
	}
	__device__ __forceinline__ vec3 brdf(const vec3& wo,const vec3& n,vec3& wi,const vec3& albedo,float& pdf,bool& delta,curandStatePhilox4_32_10_t* state) const {
		wi = bounce(wo,n,state);
		if(type == material_type::specular) {
			delta = true;
			pdf = 1;
			return albedo;
		}
		else if(type == material_type::diffuse) {
			delta = false;
			pdf = dot(n,wi) * INV_PI;
			return albedo * INV_PI;
		}
	}
};