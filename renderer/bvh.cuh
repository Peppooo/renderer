#include "scene.cuh"
#include <vector>

class box {
public:
	vec3 edge,size;
	int startIndex,length;
	__host__ bool contained(const vec3& v) {
		vec3 max = edge + size;
		return v.x <= max.x && v.y <= max.y && v.z <= max.z && v.x >= edge.x && v.y >= edge.y && v.z >= edge.z;
	}
	__host__ int count_verts(const object& obj) {
		int count = 0;
		for(int i = 0; i < 3; i++) {
			if(contained(obj[i])) {
				count++;
			}
		}
		return count;
	}
	__device__ bool intersect(const vec3& O,const vec3& D,float& dist) const {
		vec3 invDir = 1 / D;
		vec3 t0s = (edge - O) * invDir;
		vec3 t1s = (edge + size - O) * invDir;

		vec3 tmin3 = v_min(t0s,t1s);
		vec3 tmax3 = v_max(t0s,t1s);

		dist = max(max(tmin3.x,tmin3.y),tmin3.z);
		float tmax = min(min(tmax3.x,tmax3.y),tmax3.z);

		return (tmax >= max(dist,0.0f));
	}
};

class node {
public:
	box self;
	int child_nodeA = -1;
	int child_nodeB = -1;
	int depth = 0;
};

#define MAX_BVH_NODES 1000

class BVH {
private:
	node* host_pointer;
	node* device_pointer;
	bool is_pointer_host;
public:
	node* nodes;
	int nodesSize;
	__host__ void init() {
		nodes = new node[MAX_BVH_NODES];
		host_pointer = nodes;
		cudaMalloc(&device_pointer,sizeof(node) * MAX_BVH_NODES);
		nodesSize = 0;
	}
	__host__ void build(const int max_depth,object* scene,const size_t sceneSize) {
		nodesSize++;
		vec3 minb = {FLT_MAX,FLT_MAX,FLT_MAX};
		vec3 maxb = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
		for(int i = 0; i < sceneSize; i++) {
			minb = v_min(minb,v_min(v_min(scene[i].a,scene[i].b),scene[i].c));
			maxb = v_max(maxb,v_max(v_max(scene[i].a,scene[i].b),scene[i].c));
		}
		nodes[0].self = box();
		nodes[0].self.edge = minb;
		nodes[0].self.size = maxb - minb;
		nodes[0].depth = 0;
		nodes[0].self.length = sceneSize;
		nodes[0].self.startIndex = 0;
		buildChild(0,max_depth,scene,sceneSize);
	}
	__host__ void buildChild(const int idx,const int max_depth,object* scene,const size_t sceneSize) {
		nodesSize++;
		nodes[idx].child_nodeA = nodesSize - 1;
		nodesSize++;
		nodes[idx].child_nodeB = nodesSize - 1;

		int left = nodes[idx].child_nodeA;
		int right = nodes[idx].child_nodeB;

		nodes[left].depth = nodes[idx].depth + 1;
		nodes[right].depth = nodes[idx].depth + 1;

		// calcolo bounding box dei figli
		vec3 minA = {FLT_MAX,FLT_MAX,FLT_MAX};
		vec3 maxA = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
		vec3 minB = minA;
		vec3 maxB = maxA;
		int axis = max_idx(nodes[idx].self.size);
		vector<int> index_a;
		vector<int> index_b;

		vec3 newLeftSize = nodes[idx].self.size;
		newLeftSize[axis] = ceilf(newLeftSize[axis]/2.0f);
		nodes[left].self.size = newLeftSize;
		nodes[left].self.edge = nodes[idx].self.edge;

		//float leftMax = (nodes[left].self.edge + nodes[left].self.size)[axis];

		for(int i = 0; i < sceneSize; i++) {
			int numVertsIn = nodes->self.count_verts(scene[i]);
			if(numVertsIn == 0) {
				index_b.push_back(i);
			}
			else if(numVertsIn == 1 || numVertsIn == 2) {
				nodes[left].self.size[axis] = (v_max(scene[i].a,v_max(scene[i].b,scene[i].c))[axis] - nodes[left].self.edge[axis] + epsilon);
				index_a.push_back(i);
			}
			else {
				index_a.push_back(i);
			}
		}



		nodes[right].self.size = nodes[idx].self.size;
		nodes[right].self.size[axis] -= nodes[left].self.size[axis];

		nodes[right].self.edge = nodes[idx].self.edge;
		nodes[right].self.edge[axis] += nodes[left].self.size[axis];


		//nodes[left].self = box();
		//nodes[left].self.edge = minA;
		//nodes[left].self.size = maxA - minA;
		//nodes[right].self = box();
		//nodes[right].self.edge = minB;
		//nodes[right].self.size = maxB - minB;

		//nodes[left].self.contained_trigs = trisA;
		//nodes[right].self.contained_trigs = trisB;
		object* new_scene = new object[MAX_OBJ];
		//copy(scene,scene + sceneSize,new_scene); // copy scene
		copy(scene,scene+MAX_OBJ,new_scene);
		

		for(int i = 0; i < index_a.size(); i++) {
			new_scene[nodes[idx].self.startIndex + i] = scene[index_a[i]];
		}
		nodes[left].self.length = index_a.size();
		nodes[left].self.startIndex = nodes[idx].self.startIndex;

		for(int i = 0; i < index_b.size(); i++) {
			new_scene[nodes[idx].self.startIndex + i + index_a.size()] = scene[index_b[i]];
		}
		nodes[right].self.length = index_b.size();
		nodes[right].self.startIndex = nodes[idx].self.startIndex + nodes[left].self.length;

		copy(new_scene,new_scene+ MAX_OBJ,scene);



		if(nodes[left].depth < max_depth && index_a.size() > 4) {
			buildChild(left,max_depth,scene,sceneSize);
		}
		if(nodes[right].depth < max_depth && index_b.size()>4) {
			buildChild(right,max_depth,scene,sceneSize);
		}

		cudaMemcpy(device_pointer,nodes,sizeof(node) * MAX_BVH_NODES,cudaMemcpyHostToDevice);
	}
	__device__ bool castRay(const vec3& o,const vec3& d,box& box) const {

		
		float c_dist = FLT_MAX;
		float min_dist = FLT_MAX;
		bool hit = false;
		for(int i = 0; i < nodesSize; i++) {
			if(nodes[i].self.intersect(o,d,c_dist)) {
				if(nodes[i].child_nodeA == -1 && min_dist>c_dist) {
					//checks.insert(checks.end(),nodes[i].self.contained_trigs.begin(),nodes[i].self.contained_trigs.end());
					if(nodes[i].self.length > 0) {
						min_dist = c_dist;
						box = nodes[i].self;
						hit = true;
					}
				}
			}
		}
		return hit;
	}
	void convertToDevice() {
		if(is_pointer_host) {
			nodes = device_pointer;
			is_pointer_host = false;
		}
	}
	void convertToHost() {
		if(!is_pointer_host) {
			nodes = host_pointer;
			is_pointer_host = true;
		}
	}
};