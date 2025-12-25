---
sidebar_position: 5
---

# Environment Modeling

This guide covers the fundamentals of creating and configuring simulation environments for robotics applications. Environment modeling is crucial for testing robots in realistic scenarios before deploying them in the real world.

## Overview

Environment modeling in robotics simulation involves creating digital representations of the physical world where robots operate. This includes:

- **Static environments**: Buildings, rooms, obstacles
- **Dynamic elements**: Moving objects, people, other robots
- **Environmental conditions**: Lighting, weather, terrain
- **Sensor properties**: Reflectivity, texture, material properties

### Key Concepts

- **Scene Graph**: Hierarchical organization of objects in the environment
- **Coordinate Systems**: Reference frames for spatial relationships
- **Level of Detail (LOD)**: Different representations for different use cases
- **Procedural Generation**: Automated creation of complex environments
- **Realism vs. Performance**: Balancing visual fidelity with computational efficiency

## Basic Environment Components

### Coordinate Systems

Understanding coordinate systems is fundamental to environment modeling:

```python
import numpy as np

class CoordinateSystems:
    """Different coordinate systems used in robotics simulation"""

    def __init__(self):
        # World coordinate system (global reference frame)
        self.world_frame = np.eye(4)  # Identity transformation

    def world_to_robot(self, world_point, robot_pose):
        """
        Transform from world coordinates to robot coordinates
        robot_pose: 4x4 transformation matrix
        """
        robot_to_world = robot_pose
        world_to_robot = np.linalg.inv(robot_to_world)
        world_homogeneous = np.append(world_point, 1)
        robot_homogeneous = world_to_robot @ world_homogeneous
        return robot_homogeneous[:3]

    def robot_to_world(self, robot_point, robot_pose):
        """Transform from robot coordinates to world coordinates"""
        world_point = robot_pose @ np.append(robot_point, 1)
        return world_point[:3]

    def camera_to_world(self, camera_point, camera_pose):
        """Transform from camera coordinates to world coordinates"""
        return self.robot_to_world(camera_point, camera_pose)

    def world_to_camera(self, world_point, camera_pose):
        """Transform from world coordinates to camera coordinates"""
        return self.world_to_robot(world_point, camera_pose)
```

### Basic Shapes and Primitives

Start with basic geometric shapes for environment modeling:

```python
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ShapeType(Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    MESH = "mesh"

@dataclass
class Shape:
    """Basic shape definition for environment objects"""
    shape_type: ShapeType
    dimensions: np.ndarray  # [width, height, depth] for box, [radius, height] for cylinder, etc.
    mass: float = 1.0
    material: str = "default"
    friction: float = 0.5
    restitution: float = 0.1

class BasicShapes:
    """Factory for creating basic environment shapes"""

    @staticmethod
    def create_box(width, height, depth, **kwargs):
        """Create a box-shaped object"""
        dimensions = np.array([width, height, depth])
        return Shape(ShapeType.BOX, dimensions, **kwargs)

    @staticmethod
    def create_sphere(radius, **kwargs):
        """Create a spherical object"""
        dimensions = np.array([radius])
        return Shape(ShapeType.SPHERE, dimensions, **kwargs)

    @staticmethod
    def create_cylinder(radius, height, **kwargs):
        """Create a cylindrical object"""
        dimensions = np.array([radius, height])
        return Shape(ShapeType.CYLINDER, dimensions, **kwargs)

    @staticmethod
    def create_mesh(mesh_file, **kwargs):
        """Create an object from a mesh file"""
        dimensions = np.array([0])  # Dimensions determined by mesh
        return Shape(ShapeType.MESH, dimensions, **kwargs)
```

## Creating Indoor Environments

### Room Layout

Design indoor environments with proper room layouts:

```python
class IndoorEnvironment:
    """Class for creating indoor simulation environments"""

    def __init__(self, name="indoor_env"):
        self.name = name
        self.objects = []
        self.rooms = []
        self.obstacles = []

    def add_room(self, room_name, dimensions, position=[0, 0, 0]):
        """Add a room to the environment"""
        room = {
            'name': room_name,
            'dimensions': np.array(dimensions),  # [length, width, height]
            'position': np.array(position),
            'walls': [],
            'doors': [],
            'furniture': []
        }

        # Create walls automatically
        self._create_room_walls(room)

        self.rooms.append(room)
        return room

    def _create_room_walls(self, room):
        """Automatically create walls for a room"""
        length, width, height = room['dimensions']
        center_x, center_y, center_z = room['position']

        # Wall thickness
        wall_thickness = 0.2

        # Create 4 walls
        wall_positions = [
            # North wall
            {
                'position': [center_x, center_y + width/2, center_z + height/2],
                'dimensions': [length, wall_thickness, height]
            },
            # South wall
            {
                'position': [center_x, center_y - width/2, center_z + height/2],
                'dimensions': [length, wall_thickness, height]
            },
            # East wall
            {
                'position': [center_x + length/2, center_y, center_z + height/2],
                'dimensions': [wall_thickness, width, height]
            },
            # West wall
            {
                'position': [center_x - length/2, center_y, center_z + height/2],
                'dimensions': [wall_thickness, width, height]
            }
        ]

        for wall_pos in wall_positions:
            wall_shape = BasicShapes.create_box(
                wall_pos['dimensions'][0],
                wall_pos['dimensions'][1],
                wall_pos['dimensions'][2],
                material='wall',
                friction=0.8,
                restitution=0.1
            )
            room['walls'].append({
                'shape': wall_shape,
                'position': wall_pos['position']
            })

    def add_furniture(self, room_name, furniture_type, position, dimensions):
        """Add furniture to a specific room"""
        room = next((r for r in self.rooms if r['name'] == room_name), None)
        if not room:
            raise ValueError(f"Room {room_name} not found")

        furniture_shape = BasicShapes.create_box(*dimensions)
        furniture = {
            'type': furniture_type,
            'shape': furniture_shape,
            'position': np.array(position)
        }
        room['furniture'].append(furniture)
        self.objects.append(furniture)

    def add_obstacle(self, position, shape, name="obstacle"):
        """Add a static obstacle to the environment"""
        obstacle = {
            'name': name,
            'shape': shape,
            'position': np.array(position),
            'is_static': True
        }
        self.obstacles.append(obstacle)
        self.objects.append(obstacle)

# Example usage
def create_office_environment():
    """Create an office environment example"""
    env = IndoorEnvironment("office")

    # Add main office room
    env.add_room("main_office", [10, 8, 3])  # 10m x 8m x 3m

    # Add desk
    env.add_furniture("main_office", "desk", [2, 0, 0.75], [1.5, 0.8, 0.75])

    # Add chair
    env.add_furniture("main_office", "chair", [1.5, -0.5, 0.5], [0.5, 0.5, 0.5])

    # Add bookshelf
    env.add_furniture("main_office", "bookshelf", [-4, 2, 1], [0.3, 1.2, 2])

    # Add some obstacles
    env.add_obstacle([0, 2, 0.5], BasicShapes.create_sphere(0.3), "plant_pot")

    return env
```

### Creating Outdoor Environments

```python
class OutdoorEnvironment:
    """Class for creating outdoor simulation environments"""

    def __init__(self, name="outdoor_env"):
        self.name = name
        self.terrain = None
        self.obstacles = []
        self.static_objects = []
        self.weather_conditions = {}

    def create_flat_terrain(self, size_x, size_y, position=[0, 0, 0]):
        """Create a flat terrain"""
        ground_shape = BasicShapes.create_box(size_x, size_y, 0.1)  # Thin ground plane
        self.terrain = {
            'type': 'flat',
            'shape': ground_shape,
            'position': np.array([position[0], position[1], position[2] - 0.05])  # Slightly below to avoid floating
        }

    def create_rough_terrain(self, size_x, size_y, roughness=0.1):
        """Create a rough terrain with bumps"""
        # This would typically use a heightmap in practice
        self.terrain = {
            'type': 'rough',
            'size': [size_x, size_y],
            'roughness': roughness,
            'heightmap': self._generate_heightmap(size_x, size_y, roughness)
        }

    def _generate_heightmap(self, size_x, size_y, roughness, resolution=100):
        """Generate a heightmap for rough terrain"""
        x = np.linspace(0, size_x, resolution)
        y = np.linspace(0, size_y, resolution)
        xx, yy = np.meshgrid(x, y)

        # Add random noise for roughness
        heights = np.random.normal(0, roughness, (resolution, resolution))

        # Add some larger features
        heights += 0.1 * np.sin(xx * 0.5) * np.cos(yy * 0.5)

        return heights

    def add_building(self, position, dimensions, floors=1):
        """Add a building to the environment"""
        building = {
            'type': 'building',
            'position': np.array(position),
            'dimensions': np.array(dimensions),
            'floors': floors,
            'shape': BasicShapes.create_box(*dimensions)
        }
        self.static_objects.append(building)

    def add_tree(self, position, trunk_radius=0.3, trunk_height=3.0, canopy_radius=2.0):
        """Add a tree to the environment"""
        tree = {
            'type': 'tree',
            'position': np.array(position),
            'trunk': {
                'shape': BasicShapes.create_cylinder(trunk_radius, trunk_height),
                'position': position
            },
            'canopy': {
                'shape': BasicShapes.create_sphere(canopy_radius),
                'position': [position[0], position[1], position[2] + trunk_height]
            }
        }
        self.static_objects.append(tree)

    def add_street_elements(self, street_width=10, street_length=50):
        """Add typical street elements"""
        # Add road
        road = {
            'type': 'road',
            'shape': BasicShapes.create_box(street_length, street_width, 0.2),
            'position': [0, 0, -0.1],
            'material': 'asphalt'
        }
        self.static_objects.append(road)

        # Add sidewalk
        sidewalk = {
            'type': 'sidewalk',
            'shape': BasicShapes.create_box(street_length, 2, 0.1),
            'position': [0, street_width/2 + 1, -0.05],
            'material': 'concrete'
        }
        self.static_objects.append(sidewalk)

        sidewalk2 = {
            'type': 'sidewalk',
            'shape': BasicShapes.create_box(street_length, 2, 0.1),
            'position': [0, -street_width/2 - 1, -0.05],
            'material': 'concrete'
        }
        self.static_objects.append(sidewalk2)

# Example usage
def create_city_environment():
    """Create a city environment example"""
    env = OutdoorEnvironment("city")

    # Create terrain
    env.create_flat_terrain(100, 100)

    # Add buildings
    env.add_building([20, 10, 15], [20, 15, 30], floors=5)
    env.add_building([-15, -20, 12], [15, 10, 25], floors=4)

    # Add trees
    env.add_tree([5, 5, 0])
    env.add_tree([-5, -5, 0])
    env.add_tree([10, -8, 0])

    # Add street
    env.add_street_elements()

    return env
```

## Advanced Environment Features

### Procedural Environment Generation

Create environments programmatically:

```python
import random

class ProceduralEnvironmentGenerator:
    """Generate environments procedurally"""

    def __init__(self):
        self.environment_templates = {
            'warehouse': self._generate_warehouse,
            'office': self._generate_office,
            'outdoor_urban': self._generate_urban_outdoor,
            'maze': self._generate_maze
        }

    def generate_environment(self, template_name, **params):
        """Generate an environment based on template"""
        if template_name not in self.environment_templates:
            raise ValueError(f"Unknown template: {template_name}")

        generator_func = self.environment_templates[template_name]
        return generator_func(**params)

    def _generate_warehouse(self, width=50, length=30, height=10, num_shelves=10):
        """Generate a warehouse environment"""
        env = IndoorEnvironment("warehouse")

        # Create warehouse room
        env.add_room("main_area", [width, length, height])

        # Add shelves randomly
        for i in range(num_shelves):
            shelf_x = random.uniform(-width/2 + 2, width/2 - 2)
            shelf_y = random.uniform(-length/2 + 2, length/2 - 2)

            shelf_dims = [
                random.uniform(1.5, 2.5),  # width
                random.uniform(0.3, 0.5),  # depth
                random.uniform(1.5, 3.0)   # height
            ]

            env.add_furniture("main_area", "shelf", [shelf_x, shelf_y, shelf_dims[2]/2], shelf_dims)

        return env

    def _generate_office(self, num_rooms=4, room_size_range=(8, 12)):
        """Generate an office building with multiple rooms"""
        env = IndoorEnvironment("office_building")

        room_size = random.uniform(*room_size_range)
        corridor_width = 2

        for i in range(num_rooms):
            room_x = (i % 2) * (room_size + corridor_width)
            room_y = (i // 2) * (room_size + corridor_width)

            room_pos = [room_x, room_y, 0]
            env.add_room(f"room_{i}", [room_size, room_size, 3], room_pos)

            # Add office furniture
            if i % 2 == 0:  # Even-numbered rooms get desks
                env.add_furniture(f"room_{i}", "desk", [0, -1, 0.75], [1.5, 0.8, 0.75])
                env.add_furniture(f"room_{i}", "chair", [0.5, -1.5, 0.5], [0.5, 0.5, 0.5])

        return env

    def _generate_urban_outdoor(self, city_size=100, building_density=0.3):
        """Generate an urban outdoor environment"""
        env = OutdoorEnvironment("urban")

        # Create terrain
        env.create_flat_terrain(city_size, city_size)

        # Add roads
        road_spacing = 20
        for x in range(-city_size//2, city_size//2, road_spacing):
            # Horizontal road
            road_h = {
                'type': 'road',
                'shape': BasicShapes.create_box(city_size, 8, 0.2),
                'position': [0, x, -0.1],
                'material': 'asphalt'
            }
            env.static_objects.append(road_h)

        for y in range(-city_size//2, city_size//2, road_spacing):
            # Vertical road
            road_v = {
                'type': 'road',
                'shape': BasicShapes.create_box(8, city_size, 0.2),
                'position': [y, 0, -0.1],
                'material': 'asphalt'
            }
            env.static_objects.append(road_v)

        # Add buildings
        num_buildings = int(city_size * city_size * building_density / (20 * 20))
        for _ in range(num_buildings):
            x = random.randint(-city_size//2 + 10, city_size//2 - 10)
            y = random.randint(-city_size//2 + 10, city_size//2 - 10)

            # Check if position is on a road
            on_road = any(
                abs(x - road_obj['position'][0]) < 10 or
                abs(y - road_obj['position'][1]) < 10
                for road_obj in env.static_objects
                if road_obj['type'] == 'road'
            )

            if not on_road:
                building_width = random.uniform(15, 25)
                building_depth = random.uniform(15, 25)
                building_height = random.uniform(20, 50)

                env.add_building([x, y, building_height/2],
                                [building_width, building_depth, building_height],
                                floors=int(building_height/3))

        return env

    def _generate_maze(self, maze_size=20, wall_height=2, wall_thickness=0.3):
        """Generate a maze environment"""
        env = IndoorEnvironment("maze")

        # Create boundary
        env.add_room("maze_boundary", [maze_size, maze_size, wall_height])

        # Generate maze walls (simplified - would use proper maze generation algorithm in practice)
        maze_grid = self._create_maze_grid(maze_size)

        for i in range(len(maze_grid)):
            for j in range(len(maze_grid[i])):
                if maze_grid[i][j] == 1:  # Wall
                    wall_pos = [
                        -maze_size/2 + j + 0.5,
                        -maze_size/2 + i + 0.5,
                        wall_height/2
                    ]
                    wall_shape = BasicShapes.create_box(wall_thickness, wall_thickness, wall_height)
                    env.add_obstacle(wall_pos, wall_shape, f"maze_wall_{i}_{j}")

        return env

    def _create_maze_grid(self, size):
        """Create a simple maze grid (simplified version)"""
        # This is a very basic maze - in practice you'd use proper algorithms
        grid = [[0 for _ in range(size)] for _ in range(size)]

        # Add boundary walls
        for i in range(size):
            grid[0][i] = 1
            grid[size-1][i] = 1
            grid[i][0] = 1
            grid[i][size-1] = 1

        # Add some interior walls
        for i in range(2, size-2, 3):
            for j in range(2, size-2):
                if j % 4 == 0:
                    grid[i][j] = 1

        return grid

# Example usage
def create_procedural_environments():
    """Create various procedural environments"""
    generator = ProceduralEnvironmentGenerator()

    # Generate warehouse
    warehouse_env = generator.generate_environment(
        'warehouse',
        width=40,
        length=30,
        height=8,
        num_shelves=15
    )

    # Generate office building
    office_env = generator.generate_environment(
        'office',
        num_rooms=6,
        room_size_range=(10, 15)
    )

    # Generate urban environment
    urban_env = generator.generate_environment(
        'outdoor_urban',
        city_size=80,
        building_density=0.4
    )

    # Generate maze
    maze_env = generator.generate_environment(
        'maze',
        maze_size=16,
        wall_height=2
    )

    return {
        'warehouse': warehouse_env,
        'office': office_env,
        'urban': urban_env,
        'maze': maze_env
    }
```

## Environment Integration with Simulation

### Gazebo Environment Integration

Create Gazebo-compatible environment files:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="warehouse_world">
    <!-- Physics properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include default models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Warehouse floor -->
    <model name="warehouse_floor">
      <pose>0 0 -0.05 0 0 0</pose>
      <link name="floor_link">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>50 30 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>50 30 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1000</mass>
          <inertia>
            <ixx>104166.67</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>208333.33</iyy>
            <iyz>0</iyz>
            <izz>312500.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Warehouse walls -->
    <model name="north_wall">
      <pose>0 15 2 0 0 0</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>50 0.2 4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>50 0.2 4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>16.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>104166.83</iyy>
            <iyz>0</iyz>
            <izz>104183.5</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="south_wall">
      <pose>0 -15 2 0 0 0</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>50 0.2 4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>50 0.2 4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>16.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>104166.83</iyy>
            <iyz>0</iyz>
            <izz>104183.5</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="east_wall">
      <pose>25 0 2 0 0 1.57</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>30 0.2 4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>30 0.2 4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>16.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>75016.83</iyy>
            <iyz>0</iyz>
            <izz>75033.5</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="west_wall">
      <pose>-25 0 2 0 0 1.57</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>30 0.2 4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>30 0.2 4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>16.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>75016.83</iyy>
            <iyz>0</iyz>
            <izz>75033.5</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Shelves -->
    <model name="shelf_1">
      <pose>-10 -8 1 0 0 0</pose>
      <link name="shelf_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.9 0.7 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>50</mass>
          <inertia>
            <ixx>20.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>104.17</iyy>
            <iyz>0</iyz>
            <izz>125.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="shelf_2">
      <pose>10 8 1 0 0 3.14</pose>
      <link name="shelf_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.9 0.7 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>50</mass>
          <inertia>
            <ixx>20.83</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>104.17</iyy>
            <iyz>0</iyz>
            <izz>125.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Isaac Sim Environment Integration

For Isaac Sim, create USD files for environments:

```python
# Example: Creating a USD stage for Isaac Sim
import omni
from pxr import Usd, UsdGeom, Gf, Sdf

def create_isaac_sim_warehouse_stage(stage_path):
    """Create a warehouse environment in USD format for Isaac Sim"""

    # Create new stage
    stage = Usd.Stage.CreateNew(stage_path)

    # Create world
    world = UsdGeom.Xform.Define(stage, "/World")

    # Create ground plane
    ground_plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")

    # Define ground plane vertices and faces
    vertices = [
        Gf.Vec3f(-25, -15, 0),  # Bottom-left
        Gf.Vec3f(25, -15, 0),   # Bottom-right
        Gf.Vec3f(25, 15, 0),    # Top-right
        Gf.Vec3f(-25, 15, 0)    # Top-left
    ]
    face_vertex_counts = [4]
    face_vertex_indices = [0, 1, 2, 3]

    ground_plane.GetPointsAttr().Set(vertices)
    ground_plane.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    ground_plane.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Create warehouse walls
    north_wall = create_wall(stage, "/World/NorthWall",
                           position=Gf.Vec3f(0, 15, 2),
                           size=Gf.Vec3f(50, 0.2, 4))

    south_wall = create_wall(stage, "/World/SouthWall",
                           position=Gf.Vec3f(0, -15, 2),
                           size=Gf.Vec3f(50, 0.2, 4))

    east_wall = create_wall(stage, "/World/EastWall",
                          position=Gf.Vec3f(25, 0, 2),
                          size=Gf.Vec3f(0.2, 30, 4),
                          rotation=Gf.Vec3f(0, 0, 90))

    west_wall = create_wall(stage, "/World/WestWall",
                          position=Gf.Vec3f(-25, 0, 2),
                          size=Gf.Vec3f(0.2, 30, 4),
                          rotation=Gf.Vec3f(0, 0, 90))

    # Add shelves
    shelf1 = create_shelf(stage, "/World/Shelf1",
                         position=Gf.Vec3f(-10, -8, 1))
    shelf2 = create_shelf(stage, "/World/Shelf2",
                         position=Gf.Vec3f(10, 8, 1))

    # Add lighting
    add_lighting(stage)

    # Save the stage
    stage.GetRootLayer().Save()

    return stage

def create_wall(stage, path, position, size, rotation=None):
    """Create a wall in the USD stage"""
    wall = UsdGeom.Cube.Define(stage, path)
    wall.GetSizeAttr().Set(size[0])
    wall.AddTranslateOp().Set(position)

    if rotation:
        wall.AddRotateXYZOp().Set(rotation)

    return wall

def create_shelf(stage, path, position):
    """Create a shelf in the USD stage"""
    shelf = UsdGeom.Cube.Define(stage, path)
    shelf.GetSizeAttr().Set(2.0)  # 2m cube
    shelf.AddTranslateOp().Set(position)

    return shelf

def add_lighting(stage):
    """Add lighting to the scene"""
    # Add dome light
    dome_light = UsdGeom.Xform.Define(stage, "/World/Light/DomeLight")

    # Add directional light
    directional_light = UsdGeom.Xform.Define(stage, "/World/Light/DirectionalLight")

    return dome_light, directional_light

# Example usage
def create_warehouse_environment():
    """Create and save warehouse environment"""
    stage = create_isaac_sim_warehouse_stage("./warehouse.usd")
    print(f"Warehouse environment saved to: {stage.GetRootLayer().realPath}")
```

## Environmental Conditions

### Lighting Configuration

Configure lighting for realistic environments:

```python
class LightingConfiguration:
    """Configure lighting for simulation environments"""

    def __init__(self):
        self.lights = []

    def add_directional_light(self, name, direction, intensity=1.0, color=(1, 1, 1)):
        """Add a directional light (like sun)"""
        light = {
            'name': name,
            'type': 'directional',
            'direction': np.array(direction),
            'intensity': intensity,
            'color': np.array(color)
        }
        self.lights.append(light)

    def add_point_light(self, name, position, range=10.0, intensity=1.0, color=(1, 1, 1)):
        """Add a point light"""
        light = {
            'name': name,
            'type': 'point',
            'position': np.array(position),
            'range': range,
            'intensity': intensity,
            'color': np.array(color)
        }
        self.lights.append(light)

    def add_spot_light(self, name, position, direction, inner_cone=30, outer_cone=45,
                      range=10.0, intensity=1.0, color=(1, 1, 1)):
        """Add a spot light"""
        light = {
            'name': name,
            'type': 'spot',
            'position': np.array(position),
            'direction': np.array(direction),
            'inner_cone': inner_cone,
            'outer_cone': outer_cone,
            'range': range,
            'intensity': intensity,
            'color': np.array(color)
        }
        self.lights.append(light)

    def configure_indoor_lighting(self, room_dimensions, ceiling_height=3.0):
        """Configure lighting for indoor environment"""
        length, width, height = room_dimensions

        # Add ceiling lights evenly spaced
        light_spacing = 3.0  # meters between lights
        num_x = int(length / light_spacing) + 1
        num_y = int(width / light_spacing) + 1

        for i in range(num_x):
            for j in range(num_y):
                x = -length/2 + (i + 0.5) * light_spacing
                y = -width/2 + (j + 0.5) * light_spacing
                z = ceiling_height - 0.1  # slightly below ceiling

                light_name = f"ceiling_light_{i}_{j}"
                self.add_point_light(light_name, [x, y, z],
                                   range=8.0, intensity=1500,
                                   color=(0.95, 0.95, 1.0))  # Cool white

    def configure_outdoor_lighting(self, time_of_day="noon"):
        """Configure lighting for outdoor environment"""
        if time_of_day == "noon":
            # Bright overhead sunlight
            self.add_directional_light("sun", [-0.2, -1.0, -0.5], intensity=1.5)
        elif time_of_day == "morning":
            # Gentle morning light
            self.add_directional_light("sun", [-0.7, -0.7, -0.3], intensity=1.0)
        elif time_of_day == "evening":
            # Warm evening light
            self.add_directional_light("sun", [0.7, -0.7, -0.3],
                                    intensity=0.8, color=(1.0, 0.7, 0.4))
        elif time_of_day == "night":
            # Moonlight and artificial lights
            self.add_directional_light("moon", [0.1, 0.1, -1.0],
                                    intensity=0.1, color=(0.8, 0.9, 1.0))

# Example usage
def setup_environment_lighting():
    """Set up lighting for different environment types"""
    lighting = LightingConfiguration()

    # Indoor office lighting
    lighting.configure_indoor_lighting([10, 8, 3])  # 10x8x3m room

    # Outdoor lighting for daytime
    lighting.configure_outdoor_lighting("noon")

    return lighting
```

### Weather and Atmospheric Effects

Configure environmental conditions:

```python
class EnvironmentalConditions:
    """Configure weather and atmospheric effects for simulation"""

    def __init__(self):
        self.conditions = {
            'temperature': 20.0,  # Celsius
            'humidity': 50.0,     # Percent
            'wind_speed': 0.0,    # m/s
            'wind_direction': 0.0, # degrees
            'visibility': 100.0,  # meters
            'precipitation': 0.0, # mm/hour
            'fog_density': 0.0    # 0.0 to 1.0
        }

    def set_weather_preset(self, preset_name):
        """Apply a predefined weather condition"""
        presets = {
            'clear_day': {
                'temperature': 22.0,
                'humidity': 40.0,
                'wind_speed': 2.0,
                'visibility': 100.0,
                'precipitation': 0.0,
                'fog_density': 0.0
            },
            'rainy': {
                'temperature': 15.0,
                'humidity': 90.0,
                'wind_speed': 5.0,
                'visibility': 20.0,
                'precipitation': 5.0,
                'fog_density': 0.3
            },
            'foggy': {
                'temperature': 12.0,
                'humidity': 95.0,
                'wind_speed': 1.0,
                'visibility': 10.0,
                'precipitation': 0.5,
                'fog_density': 0.7
            },
            'snowy': {
                'temperature': -2.0,
                'humidity': 80.0,
                'wind_speed': 3.0,
                'visibility': 15.0,
                'precipitation': 3.0,
                'fog_density': 0.2
            }
        }

        if preset_name in presets:
            self.conditions.update(presets[preset_name])

    def apply_to_simulation(self, sim_engine):
        """Apply environmental conditions to simulation engine"""
        if hasattr(sim_engine, 'set_temperature'):
            sim_engine.set_temperature(self.conditions['temperature'])

        if hasattr(sim_engine, 'set_wind'):
            sim_engine.set_wind(
                self.conditions['wind_speed'],
                self.conditions['wind_direction']
            )

        if hasattr(sim_engine, 'set_atmosphere'):
            sim_engine.set_atmosphere(
                fog_density=self.conditions['fog_density'],
                visibility=self.conditions['visibility']
            )

    def add_sensors_for_conditions(self, robot):
        """Add sensors to measure environmental conditions"""
        sensors = []

        # Temperature sensor
        temp_sensor = {
            'type': 'temperature',
            'name': 'temperature_sensor',
            'position': [0, 0, 0.5],
            'noise': 0.5  # degrees
        }
        sensors.append(temp_sensor)

        # Humidity sensor
        humidity_sensor = {
            'type': 'humidity',
            'name': 'humidity_sensor',
            'position': [0, 0, 0.5],
            'noise': 2.0  # percent
        }
        sensors.append(humidity_sensor)

        # Barometric pressure sensor (related to altitude/conditions)
        pressure_sensor = {
            'type': 'barometer',
            'name': 'pressure_sensor',
            'position': [0, 0, 0.5],
            'noise': 100  # Pa
        }
        sensors.append(pressure_sensor)

        # Add sensors to robot
        for sensor in sensors:
            robot.add_sensor(sensor)

        return sensors

# Example usage
def setup_environmental_conditions():
    """Set up environmental conditions for simulation"""
    env_conditions = EnvironmentalConditions()

    # Set to clear day conditions
    env_conditions.set_weather_preset('clear_day')

    return env_conditions
```

## Terrain and Surface Properties

### Creating Realistic Terrains

```python
class TerrainGenerator:
    """Generate realistic terrains for outdoor environments"""

    def __init__(self):
        self.terrain_types = {
            'flat': self._generate_flat_terrain,
            'hilly': self._generate_hilly_terrain,
            'mountainous': self._generate_mountainous_terrain,
            'urban': self._generate_urban_terrain,
            'forest': self._generate_forest_terrain
        }

    def generate_terrain(self, terrain_type, size_x, size_y, resolution=100):
        """Generate terrain based on type"""
        if terrain_type not in self.terrain_types:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

        generator = self.terrain_types[terrain_type]
        return generator(size_x, size_y, resolution)

    def _generate_flat_terrain(self, size_x, size_y, resolution):
        """Generate flat terrain"""
        x = np.linspace(-size_x/2, size_x/2, resolution)
        y = np.linspace(-size_y/2, size_y/2, resolution)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)  # Flat surface at z=0

        return {
            'type': 'flat',
            'heightmap': zz,
            'coordinates': (xx, yy, zz),
            'texture': 'grass'
        }

    def _generate_hilly_terrain(self, size_x, size_y, resolution):
        """Generate hilly terrain with natural-looking hills"""
        x = np.linspace(-size_x/2, size_x/2, resolution)
        y = np.linspace(-size_y/2, size_y/2, resolution)
        xx, yy = np.meshgrid(x, y)

        # Create multiple overlapping hills
        heights = np.zeros_like(xx)

        # Add several hills of different sizes
        for _ in range(5):
            hill_x = random.uniform(-size_x/3, size_x/3)
            hill_y = random.uniform(-size_y/3, size_y/3)
            hill_size = random.uniform(size_x/8, size_x/4)
            hill_height = random.uniform(1, 3)

            dist = np.sqrt((xx - hill_x)**2 + (yy - hill_y)**2)
            hill = hill_height * np.exp(-(dist**2) / (2 * hill_size**2))
            heights += hill

        # Add some noise for natural variation
        noise = np.random.normal(0, 0.1, heights.shape)
        heights += noise

        return {
            'type': 'hilly',
            'heightmap': heights,
            'coordinates': (xx, yy, heights),
            'texture': 'grass_dirt_mix'
        }

    def _generate_mountainous_terrain(self, size_x, size_y, resolution):
        """Generate mountainous terrain with peaks and valleys"""
        x = np.linspace(-size_x/2, size_x/2, resolution)
        y = np.linspace(-size_y/2, size_y/2, resolution)
        xx, yy = np.meshgrid(x, y)

        # Create mountain ranges
        heights = np.zeros_like(xx)

        # Add major peaks
        for _ in range(3):
            peak_x = random.uniform(-size_x/2, size_x/2)
            peak_y = random.uniform(-size_y/2, size_y/2)
            peak_height = random.uniform(5, 15)
            peak_width = random.uniform(size_x/10, size_x/6)

            dist = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2)
            peak = peak_height * np.exp(-(dist**2) / (2 * peak_width**2))
            heights = np.maximum(heights, peak)

        # Add ridges
        for _ in range(2):
            ridge_center = random.uniform(-size_x/4, size_x/4)
            ridge_width = random.uniform(size_x/8, size_x/6)
            ridge_height = random.uniform(3, 8)

            ridge = ridge_height * np.exp(-((xx - ridge_center)**2) / (2 * ridge_width**2))
            heights += ridge

        # Add valleys (negative features)
        for _ in range(2):
            valley_x = random.uniform(-size_x/3, size_x/3)
            valley_y = random.uniform(-size_y/3, size_y/3)
            valley_width = random.uniform(size_x/8, size_x/6)
            valley_depth = random.uniform(1, 3)

            dist = np.sqrt((xx - valley_x)**2 + (yy - valley_y)**2)
            valley = -valley_depth * np.exp(-(dist**2) / (2 * valley_width**2))
            heights += valley

        # Add realistic noise
        noise = np.random.normal(0, 0.5, heights.shape)
        heights += noise

        # Ensure no negative terrain below sea level
        heights = np.maximum(heights, 0)

        return {
            'type': 'mountainous',
            'heightmap': heights,
            'coordinates': (xx, yy, heights),
            'texture': 'rock_grass_mix'
        }

    def _generate_urban_terrain(self, size_x, size_y, resolution):
        """Generate urban terrain with roads and buildings"""
        x = np.linspace(-size_x/2, size_x/2, resolution)
        y = np.linspace(-size_y/2, size_y/2, resolution)
        xx, yy = np.meshgrid(x, y)

        # Start with flat terrain
        heights = np.zeros_like(xx)

        # Add roads (slightly raised)
        road_width = 8.0
        road_spacing = 20.0

        # Horizontal roads
        for road_y in np.arange(-size_y/2, size_y/2, road_spacing):
            road_mask = np.abs(yy - road_y) < road_width/2
            heights[road_mask] = 0.1  # Roads slightly raised

        # Vertical roads
        for road_x in np.arange(-size_x/2, size_x/2, road_spacing):
            road_mask = np.abs(xx - road_x) < road_width/2
            heights[road_mask] = 0.1

        # Add sidewalks (raised areas next to roads)
        sidewalk_width = 2.0
        for road_y in np.arange(-size_y/2, size_y/2, road_spacing):
            # Sidewalks on north side of horizontal roads
            sidewalk_mask = (np.abs(yy - (road_y + road_width/2 + sidewalk_width/2)) < sidewalk_width/2) & \
                           (np.abs(xx) < size_x/2)
            heights[sidewalk_mask] = 0.15

            # Sidewalks on south side of horizontal roads
            sidewalk_mask = (np.abs(yy - (road_y - road_width/2 - sidewalk_width/2)) < sidewalk_width/2) & \
                           (np.abs(xx) < size_x/2)
            heights[sidewalk_mask] = 0.15

        return {
            'type': 'urban',
            'heightmap': heights,
            'coordinates': (xx, yy, heights),
            'texture': 'asphalt_concrete_mix'
        }

    def _generate_forest_terrain(self, size_x, size_y, resolution):
        """Generate forest terrain with natural features"""
        x = np.linspace(-size_x/2, size_x/2, resolution)
        y = np.linspace(-size_y/2, size_y/2, resolution)
        xx, yy = np.meshgrid(x, y)

        # Base terrain with gentle slopes
        heights = 0.5 * np.sin(xx * 0.1) * np.cos(yy * 0.1)

        # Add patches of higher ground (clearings)
        for _ in range(5):
            patch_x = random.uniform(-size_x/3, size_x/3)
            patch_y = random.uniform(-size_y/3, size_y/3)
            patch_radius = random.uniform(size_x/10, size_x/6)
            patch_height = random.uniform(0.2, 0.5)

            dist = np.sqrt((xx - patch_x)**2 + (yy - patch_y)**2)
            patch = patch_height * np.exp(-(dist**2) / (2 * patch_radius**2))
            heights += patch

        # Add streams (low areas)
        for _ in range(2):
            stream_x = random.uniform(-size_x/3, size_x/3)
            stream_y = random.uniform(-size_y/3, size_y/3)
            stream_direction = random.uniform(0, 2*np.pi)
            stream_width = random.uniform(1, 3)

            # Create curved stream path
            stream_curve = 0.1
            for i in range(resolution//10):
                curve_offset = stream_curve * np.sin(i * 0.2)
                stream_path_x = stream_x + i * np.cos(stream_direction) * 0.5 + curve_offset * np.sin(stream_direction)
                stream_path_y = stream_y + i * np.sin(stream_direction) * 0.5 - curve_offset * np.cos(stream_direction)

                stream_mask = (xx - stream_path_x)**2 + (yy - stream_path_y)**2 < (stream_width/2)**2
                heights = np.where(stream_mask, np.minimum(heights, -0.2), heights)

        # Add natural noise
        noise = np.random.normal(0, 0.05, heights.shape)
        heights += noise

        return {
            'type': 'forest',
            'heightmap': heights,
            'coordinates': (xx, yy, heights),
            'texture': 'grass_leaf_litter_mix'
        }

# Example usage
def create_varied_terrains():
    """Create different types of terrains"""
    terrain_gen = TerrainGenerator()

    terrains = {}

    # Create different terrain types
    terrains['flat'] = terrain_gen.generate_terrain('flat', 50, 50, 100)
    terrains['hilly'] = terrain_gen.generate_terrain('hilly', 50, 50, 100)
    terrains['mountainous'] = terrain_gen.generate_terrain('mountainous', 100, 100, 150)
    terrains['urban'] = terrain_gen.generate_terrain('urban', 80, 80, 120)
    terrains['forest'] = terrain_gen.generate_terrain('forest', 60, 60, 100)

    return terrains
```

## Performance Optimization

### Level of Detail (LOD)

Implement LOD for complex environments:

```python
class LevelOfDetailManager:
    """Manage different levels of detail for environment objects"""

    def __init__(self):
        self.lod_levels = {
            'high': 1.0,    # Full detail
            'medium': 0.5,  # Reduced polygons
            'low': 0.2,     # Simplified geometry
            'minimum': 0.1  # Very simplified
        }

    def generate_lod_models(self, base_model, lod_levels=None):
        """Generate different LOD versions of a model"""
        if lod_levels is None:
            lod_levels = self.lod_levels

        lod_models = {}

        for lod_name, reduction_factor in lod_levels.items():
            if lod_name == 'high':
                # Use original model
                lod_models[lod_name] = base_model
            else:
                # Generate simplified version
                lod_models[lod_name] = self._simplify_model(base_model, reduction_factor)

        return lod_models

    def _simplify_model(self, model, reduction_factor):
        """Simplify a model by reducing polygon count"""
        # This is a conceptual implementation
        # In practice, you'd use mesh simplification algorithms
        simplified_model = {
            'vertices': model['vertices'][::int(1/reduction_factor)],
            'faces': self._reduce_faces(model['faces'], reduction_factor),
            'textures': model['textures'],  # Keep textures for visual quality
            'collision': self._approximate_collision(model['collision'], reduction_factor)
        }

        return simplified_model

    def _reduce_faces(self, faces, reduction_factor):
        """Reduce number of faces in mesh"""
        # Simplified face reduction (in practice, use proper algorithms)
        keep_every_n = max(1, int(1/reduction_factor))
        return faces[::keep_every_n]

    def _approximate_collision(self, collision_geom, reduction_factor):
        """Approximate collision geometry"""
        # For performance, approximate complex collision with simpler shapes
        if reduction_factor < 0.3:
            # Use bounding box approximation
            return self._get_bounding_box(collision_geom)
        else:
            # Use reduced version of original collision
            return self._simplify_collision(collision_geom, reduction_factor)

    def select_lod_level(self, distance, max_distance=100):
        """Select appropriate LOD level based on distance"""
        lod_ratio = distance / max_distance

        if lod_ratio < 0.1:
            return 'high'
        elif lod_ratio < 0.3:
            return 'medium'
        elif lod_ratio < 0.6:
            return 'low'
        else:
            return 'minimum'

    def apply_lod_to_environment(self, environment, viewer_position):
        """Apply LOD to environment based on viewer position"""
        for obj in environment.objects:
            distance = np.linalg.norm(obj['position'] - viewer_position)
            lod_level = self.select_lod_level(distance)

            # Use appropriate LOD model
            if hasattr(obj, 'lod_models'):
                obj['current_model'] = obj['lod_models'][lod_level]

        return environment

# Example usage
def setup_lod_environment():
    """Set up environment with LOD management"""
    lod_manager = LevelOfDetailManager()

    # Create a detailed model (conceptual)
    detailed_model = {
        'vertices': np.random.rand(1000, 3),  # 1000 vertices
        'faces': np.random.rand(2000, 3),     # 2000 faces
        'textures': 'detailed_texture.png',
        'collision': 'detailed_collision_mesh'
    }

    # Generate LOD versions
    lod_models = lod_manager.generate_lod_models(detailed_model)

    # Example object with LOD
    tree_object = {
        'type': 'tree',
        'position': np.array([10, 5, 0]),
        'lod_models': lod_models,
        'current_model': lod_models['high']  # Start with high detail
    }

    return tree_object, lod_manager
```

## Environment Validation and Testing

### Validating Environment Models

```python
class EnvironmentValidator:
    """Validate environment models for correctness and performance"""

    def __init__(self):
        self.validation_results = []

    def validate_indoor_environment(self, env):
        """Validate indoor environment for common issues"""
        results = {
            'errors': [],
            'warnings': [],
            'performance_notes': []
        }

        # Check room integrity
        for room in env.rooms:
            if not self._validate_room_integrity(room):
                results['errors'].append(f"Room {room['name']} has invalid geometry")

        # Check for object intersections
        intersections = self._find_object_intersections(env.objects)
        if intersections:
            results['errors'].extend([f"Objects intersecting: {pair}" for pair in intersections])

        # Check for unreachable areas
        unreachable_areas = self._find_unreachable_areas(env)
        if unreachable_areas:
            results['warnings'].append(f"Found {len(unreachable_areas)} unreachable areas")

        # Performance checks
        if len(env.objects) > 100:
            results['performance_notes'].append("High object count may impact performance")

        return results

    def _validate_room_integrity(self, room):
        """Validate that room has proper walls and no gaps"""
        # Check if room has 4 walls (for rectangular room)
        if len(room['walls']) != 4:
            return False

        # Check wall connectivity (simplified check)
        wall_positions = [w['position'] for w in room['walls']]
        # More complex validation would check actual wall connections

        return True

    def _find_object_intersections(self, objects):
        """Find pairs of objects that intersect"""
        intersections = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                if self._objects_intersect(obj1, obj2):
                    intersections.append((obj1.get('name', f'obj_{i}'),
                                        obj2.get('name', f'obj_{j}')))

        return intersections

    def _objects_intersect(self, obj1, obj2):
        """Check if two objects intersect"""
        # Simplified intersection check for boxes
        pos1 = obj1['position']
        dims1 = obj1['shape'].dimensions

        pos2 = obj2['position']
        dims2 = obj2['shape'].dimensions

        # Check for overlap in each dimension
        x_overlap = abs(pos1[0] - pos2[0]) < (dims1[0]/2 + dims2[0]/2)
        y_overlap = abs(pos1[1] - pos2[1]) < (dims1[1]/2 + dims2[1]/2)
        z_overlap = abs(pos1[2] - pos2[2]) < (dims1[2]/2 + dims2[2]/2)

        return x_overlap and y_overlap and z_overlap

    def _find_unreachable_areas(self, env):
        """Find areas that are unreachable by a robot"""
        # This would involve pathfinding algorithms in practice
        # Simplified version just checks for very small spaces
        unreachable = []

        # Check for very narrow passages
        for obj in env.objects:
            if obj['shape'].shape_type == ShapeType.BOX:
                dims = obj['shape'].dimensions
                min_dimension = min(dims)
                if min_dimension < 0.1:  # Less than 10cm
                    unreachable.append(obj)

        return unreachable

    def validate_outdoor_environment(self, env):
        """Validate outdoor environment"""
        results = {
            'errors': [],
            'warnings': [],
            'performance_notes': []
        }

        # Check terrain validity
        if env.terrain is None:
            results['errors'].append("No terrain defined")

        # Check for proper ground coverage
        if env.terrain and env.terrain['type'] == 'flat':
            terrain_size = env.terrain['shape'].dimensions[:2]
            static_extent = self._calculate_static_object_extent(env.static_objects)

            if static_extent[0] > terrain_size[0] or static_extent[1] > terrain_size[1]:
                results['warnings'].append("Static objects extend beyond terrain boundaries")

        # Check building spacing
        building_spacing_issues = self._check_building_spacing(env.static_objects)
        results['warnings'].extend(building_spacing_issues)

        return results

    def _calculate_static_object_extent(self, static_objects):
        """Calculate the extent of static objects"""
        if not static_objects:
            return [0, 0]

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for obj in static_objects:
            pos = obj['position']
            dims = obj['dimensions'] if 'dimensions' in obj else [1, 1, 1]

            min_x = min(min_x, pos[0] - dims[0]/2)
            max_x = max(max_x, pos[0] + dims[0]/2)
            min_y = min(min_y, pos[1] - dims[1]/2)
            max_y = max(max_y, pos[1] + dims[1]/2)

        return [max_x - min_x, max_y - min_y]

    def _check_building_spacing(self, static_objects):
        """Check if buildings are properly spaced"""
        warnings = []
        buildings = [obj for obj in static_objects if obj['type'] == 'building']

        for i, building1 in enumerate(buildings):
            for j, building2 in enumerate(buildings[i+1:], i+1):
                dist = np.linalg.norm(
                    np.array(building1['position'][:2]) -
                    np.array(building2['position'][:2])
                )

                min_dist = (building1['dimensions'][0] + building1['dimensions'][1] +
                           building2['dimensions'][0] + building2['dimensions'][1]) / 4

                if dist < min_dist * 0.5:  # Too close
                    warnings.append(f"Buildings {i} and {j} are too close together")

        return warnings

    def run_full_validation(self, environment):
        """Run comprehensive validation on environment"""
        if hasattr(environment, 'rooms'):  # Indoor environment
            return self.validate_indoor_environment(environment)
        else:  # Outdoor environment
            return self.validate_outdoor_environment(environment)

# Example usage
def validate_environment_setup():
    """Validate environment setup"""
    validator = EnvironmentValidator()

    # Create test environment
    env = create_office_environment()

    # Run validation
    results = validator.run_full_validation(env)

    # Print results
    print("Environment Validation Results:")
    print(f"Errors: {len(results['errors'])}")
    for error in results['errors']:
        print(f"  - {error}")

    print(f"Warnings: {len(results['warnings'])}")
    for warning in results['warnings']:
        print(f"  - {warning}")

    print(f"Performance Notes: {len(results['performance_notes'])}")
    for note in results['performance_notes']:
        print(f"  - {note}")

    return results
```

## Best Practices

### 1. Environment Design Principles

- **Realism vs. Performance**: Balance visual fidelity with computational efficiency
- **Modularity**: Design environments in reusable components
- **Scalability**: Create environments that can be easily modified
- **Consistency**: Maintain consistent coordinate systems and units

### 2. Performance Considerations

- **Polygon Count**: Limit total polygon count for real-time performance
- **Texture Resolution**: Use appropriate texture sizes
- **Object Density**: Avoid overcrowding environments
- **LOD Implementation**: Use level-of-detail for distant objects

### 3. Validation and Testing

- **Intersection Checks**: Ensure objects don't intersect
- **Reachability**: Verify all areas are accessible
- **Physics Validation**: Test with simulation physics
- **Sensor Simulation**: Validate sensor readings in environment

### 4. Documentation and Maintenance

- **Environment Specifications**: Document coordinate systems and units
- **Object Properties**: Record physical properties for each object
- **Change Tracking**: Maintain version control for environment files
- **Reproducibility**: Ensure environments can be recreated consistently

## Next Steps

After mastering environment modeling:

1. Continue to [Week 5: Perception Systems](../week-05/index.md) to learn about sensor integration
2. Practice creating various types of environments for your robot
3. Test your environments with different robots and scenarios
4. Optimize environments for performance and realism

Your understanding of environment modeling is now foundational for creating realistic and functional simulation environments in the Physical AI and Humanoid Robotics course!