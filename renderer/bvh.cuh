#include "scene.cuh"
#include <vector>
#include <iostream>

__device__ void d_swap(float& f1,float& f2) {
	float copy = f1;
	f1 = f2;
	f2 = copy;
}

struct box {
	vec3 Min,Max;
	int startIndex; int trigCount;
	__host__ __device__ vec3 center() const {
		return (Min + Max) * 0.5f;
	}
	float surf_area() const {
		vec3 L = Max - Min;
		return L.x*L.x+L.y*L.y+L.z*L.z;
	}
	void reset(int startIdx=0) {
		Min = vec3::One * INFINITY;
		Max = vec3::One * -INFINITY;
		startIndex = startIdx;
		trigCount = 0;
	}
	__device__ bool intersect(const vec3& O,const vec3& D,float& dist) const
	{
		vec3 invDir = 1 / D;
		vec3 tMin = (Min - O)*invDir;
		vec3 tMax = (Max - O) * invDir;
		vec3 t1 = v_min(tMin,tMax);
		vec3 t2 = v_max(tMin,tMax);
		float dstFar = min(min(t2.x,t2.y),t2.z);
		float dstNear = max(max(t1.x,t1.y),t1.z);
		if(dstFar >= dstNear && dstFar > 0) { dist = dstNear; return true; };
		return false;
	}
	void grow_to_include_point(const vec3& v) {
		Max = v_max(v,Max);
		Min = v_min(v,Min);
	}
	void grow_to_include(const object& t) {
		grow_to_include_point(t.a);
		grow_to_include_point(t.b);
		grow_to_include_point(t.c);
	}
	bool contain_point(const vec3& v) const {
		return v.x<=Max.x&&v.y<=Max.y&&v.z<=Max.z&& Min.x <= v.x && Min.y <= v.y && Min.z <= v.z;
	}
	bool contain(const object& o) const {
		return contain_point(o.a) && contain_point(o.b) && contain_point(o.c);
	}
};


class node {
public:
	box bounds;
	int leftChild = 0;
	int rightChild = 0;
};
#define max_nodes MAX_OBJ



class bvh {
private:
	node* dev_nodes;
public:
	node* nodes = new node[max_nodes];
	int nodesCount=0;
	void buildChildren(object* scene,
		int sceneSize,
		int idx,
		int currentDepth,
		int maxDepth)
	{
		const box& parentBounds = nodes[idx].bounds;
		const int start = parentBounds.startIndex;
		const int count = parentBounds.trigCount;

		if(count <= 4 || currentDepth >= maxDepth)
			return;

		vec3 extent = parentBounds.Max - parentBounds.Min;

		float bestCost = INFINITY;
		int   bestAxis = -1;
		float bestSplitPos = 0.0f;

		box bestLeftBounds,bestRightBounds;
		int bestLeftCount = 0,bestRightCount = 0;

		for(int ax = 0; ax < 3; ax++) {
			if(extent[ax] <= 0.0f)
				continue;

			for(int b = 1; b < 16; b++) {
				float t = b / 16.0f;
				float splitPos = parentBounds.Min.axis(ax) + extent[ax] * t;

				box leftBox,rightBox;
				leftBox.reset();
				rightBox.reset();

				int leftCount = 0;
				int rightCount = 0;

				for(int i = 0; i < count; i++) {
					int objIdx = start + i;
					float c = scene[objIdx].center()[ax];

					if(c < splitPos) {
						leftBox.grow_to_include(scene[objIdx]);
						leftCount++;
					}
					else {
						rightBox.grow_to_include(scene[objIdx]);
						rightCount++;
					}
				}

				if(leftCount == 0 || rightCount == 0)
					continue;

				float cost =
					leftBox.surf_area() * leftCount +
					rightBox.surf_area() * rightCount;

				if(cost < bestCost) {
					bestCost = cost;
					bestAxis = ax;
					bestSplitPos = splitPos;
					bestLeftBounds = leftBox;
					bestRightBounds = rightBox;
					bestLeftCount = leftCount;
					bestRightCount = rightCount;
				}
			}
		}

		if(bestAxis == -1) 
			return;

		int leftChild = nodesCount++;
		int rightChild = nodesCount++;

		nodes[idx].leftChild = leftChild;
		nodes[idx].rightChild = rightChild;

		int i = start;
		int j = start + count - 1;

		while(i <= j) {
			if(scene[i].center()[bestAxis] < bestSplitPos) {
				i++;
			}
			else {
				swap(scene[i],scene[j]);
				j--;
			}
		}

		nodes[leftChild].bounds = bestLeftBounds;
		nodes[leftChild].bounds.startIndex = start;
		nodes[leftChild].bounds.trigCount = bestLeftCount;

		nodes[rightChild].bounds = bestRightBounds;
		nodes[rightChild].bounds.startIndex = start + bestLeftCount;
		nodes[rightChild].bounds.trigCount = bestRightCount;

		// === Recurse ===
		if(nodesCount < max_nodes - 2) {
			buildChildren(scene,sceneSize,leftChild,
				currentDepth + 1,maxDepth);
			buildChildren(scene,sceneSize,rightChild,
				currentDepth + 1,maxDepth);
		}
		else {
			cout << "BVH building stopped by nodes out of bounds" << endl;
		}
	}
	void build(const int max_depth,object* scene,const int sceneSize) {
		nodesCount = 0;
		int root = nodesCount++;
		nodes[root].bounds.Min = vec3::Zero;
		nodes[root].bounds.Max = vec3::Zero;
		nodes[root].bounds.trigCount = 0;
		nodes[root].bounds.startIndex = 0;
		for(int i = 0; i < sceneSize; i++) {
			nodes[root].bounds.grow_to_include(scene[i]);
			nodes[root].bounds.trigCount++;
		}
		cout << "building bvh structure.." << endl;
		buildChildren(scene,sceneSize,root,0,max_depth);
		cudaMalloc(&dev_nodes,sizeof(node) * nodesCount);
		cudaMemcpy(dev_nodes,nodes,sizeof(node) * nodesCount,cudaMemcpyHostToDevice);
	}
	__device__ int castRay(const Scene* scene,const vec3& o,const vec3& d,vec3& p,vec3& n,int* debug=nullptr) const {
		int stack[64]; int stackSize = 0;
		stack[stackSize++] = 0;
		float min_dist = INFINITY;
		float current_dist = INFINITY;
		int hitIdx = -1;
		if(debug) *debug = 0;
		// start from root
		while(stackSize > 0) {
			stackSize--;
			const node current_node = dev_nodes[stack[stackSize]];
			// if leaf node, add indecies
			if(current_node.leftChild == 0 && current_node.rightChild == 0) {
				// iterate triangles
				for(int i = 0; i < current_node.bounds.trigCount; i++) {
					vec3 _p,_n;
					if(debug) (*debug)++;
					if(scene->intersect(i + current_node.bounds.startIndex,o,d,_p,_n)) {
						current_dist = (_p - o).len2();
						if(current_dist < min_dist) {
							p = _p,n = _n;
							min_dist = current_dist;
							hitIdx = i + current_node.bounds.startIndex;
						}
					}
				}
			}
			else {
				// push children to stack
				float distLeft,distRight;
				bool hitLeft = dev_nodes[current_node.leftChild].bounds.intersect(o,d,distLeft);
				bool hitRight = dev_nodes[current_node.rightChild].bounds.intersect(o,d,distRight);
				distLeft = distLeft * distLeft;
				distRight = distRight * distRight;

				if(distLeft < distRight) {
					if(hitLeft && distLeft < min_dist) stack[stackSize++] = current_node.leftChild;
					if(hitRight && distRight < min_dist) stack[stackSize++] = current_node.rightChild;
				}
				else {
					if(hitRight && distRight < min_dist) stack[stackSize++] = current_node.rightChild;
					if(hitLeft && distLeft < min_dist) stack[stackSize++] = current_node.leftChild;
				}
			}
		}
		return hitIdx;
	}
	__device__ bool castRayShadow(const Scene* scene,const vec3& o,const vec3& d,const vec3& L) const {
		int stack[64]; int stackSize = 0;
		stack[stackSize++] = 0;
		float min_dist = INFINITY;
		float current_dist = INFINITY;
		// start from root
		float max_dist = (o - L).len();
		while(stackSize > 0) {
			int current_idx = stack[stackSize - 1]; stackSize--;
			float bound_dist = 0;
			if(dev_nodes[current_idx].bounds.intersect(o,d,bound_dist) && bound_dist < max_dist) {
				// if leaf node, add indecies
				if(dev_nodes[current_idx].leftChild == 0 && dev_nodes[current_idx].rightChild == 0) {
					// iterate triangles
					for(int i = 0; i < dev_nodes[current_idx].bounds.trigCount; i++) {
						vec3 _p,_n;
						if(scene->intersect(i + dev_nodes[current_idx].bounds.startIndex,o,d,_p,_n) && (_p - o).len() < max_dist) {
							return false;
						}
					}
				}
				else {
					// push children to stack
					if((dev_nodes[dev_nodes[current_idx].leftChild].bounds.center() - o).len2() < (dev_nodes[dev_nodes[current_idx].rightChild].bounds.center() - o).len2()) {
						if(dev_nodes[current_idx].leftChild != 0)
							stack[stackSize++] = (dev_nodes[current_idx].leftChild);
						if(dev_nodes[current_idx].rightChild != 0)
							stack[stackSize++] = (dev_nodes[current_idx].rightChild);
					}
					else {
						if(dev_nodes[current_idx].rightChild != 0)
							stack[stackSize++] = (dev_nodes[current_idx].leftChild);
						if(dev_nodes[current_idx].leftChild != 0)
							stack[stackSize++] = (dev_nodes[current_idx].rightChild);
					}

				}
			}
		}
		return true;
	}

	void printNodes()
	{

		for(int i = 0; i < nodesCount; i++) {
			if(nodes[i].leftChild == 0 && nodes[i].rightChild == 0) {
				cout << "Length: " << nodes[i].bounds.trigCount << " , childA,B: " << nodes[i].leftChild << " , " << nodes[i].rightChild << endl;
			}
		}
	}
};