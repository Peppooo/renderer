#include "algebra.cuh"

enum class material_type : uint8_t {
	diffuse,
	specular,
	glossy
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
	uint8_t emission;
	material() {};
	material(material_type Type,uint8_t Emission = 0): type(Type),emission(Emission) {};
	__device__ bool needs_sampling() const {
		return type == material_type::glossy;
	}
	__device__ __forceinline__ vec3 bounce(const vec3& D,vec3 n,curandStatePhilox4_32_10_t* state) const {
		if(type == material_type::specular) {
			return D-n*2*dot(D,n);
		}
		/*else if(type == glossy) {
			if(dot(D,n) > 0) n = -n;
			vec3 specular = (D - n * 2 * dot(D,n)).norm();
			
			vec3 lat1 = any_perpendicular(specular); // computes the two perpendiculars to specular 
			vec3 lat2 = cross(specular,lat1);

			lat1 = (lat1 * shininess + specular * (1 - shininess)).norm();
			lat2 = (lat2 * shininess + specular * (1 - shininess)).norm();

			vec3 lat3 = ((-lat1) * shininess + specular * (1 - shininess)).norm();
			vec3 lat4 = ((-lat2) * shininess + specular * (1 - shininess)).norm();

			recompute:
			float weights[4];
			randomList(weights,4,state);
			float _sum = sum(weights,4);
			
			vec3 result = ((lat1 * weights[0]+lat2*weights[1]+lat3*weights[2]+lat4*weights[3]) / _sum);
			if(dot(result,n) < 0) {
				goto recompute;
			}
			return result; // stupid glossy reflection i made at the start of the project
		}*/
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