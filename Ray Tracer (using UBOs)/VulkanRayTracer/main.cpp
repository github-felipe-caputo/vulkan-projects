#include <vulkan/vulkan.h>

#include "lodepng.h" // Image helper lib

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp> // Some helper functions and classes

#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>
#include <set>
#include <fstream>
#include <cmath>

// Window values
const int WINDOW_WIDTH = 500;
const int WINDOW_HEIGHT = 500;
const int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.

#ifdef DEBUG
    const bool enableValidationLayers = true;
#else
    const bool enableValidationLayers = false;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

struct QueueFamilyIndices {
    int computeFamily = -1;
    
    bool isComplete() {
        return computeFamily >= 0;
    }
};

// Proxy function for vkCreateDebugUtilsMessengerEXT
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pCallback) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Proxy function for vkDestroyDebugUtilsMessengerEXT
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT callback,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}

// Helper function to read shader files
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }
    
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

class RenderTriangle {
public:
    RenderTriangle() {
    }
    
    void run() {
        initVulkan();
        cleanup();
    }
private:
    // Compute shader will render a scene using RGBA values
    struct Pixel {
        float r, g, b, a;
    };
    
    struct Sphere {
        float radius;
        glm::vec3 position;
        glm::vec3 albedo;
        glm::vec3 specular;
    };
    
    // To use Vulkanm you need an instance
    VkInstance instance;
    
    // This is the debugger callback
    VkDebugUtilsMessengerEXT callback;
    
    // To setup up vulkan we also need
    // VkPhysicalDevice: some device on the computer that support vulkan
    // VkDevice: a "logical" device, that we use to interact with the physical one
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    
    // To execute commands on vulkan, you need to send them to a queue
    // (from the command buffer to the queue)
    // We also need to set up all queues we need for the program
    VkQueue computeQueue;
    
    // Descriptors are used so we can utilize resources on shaders
    // A descriptorPool will allocate the resource we want (for instance, uniform buffers)
    // And the descriptors are all added in the set
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
    // This will be where the compute shader will store it's result
    // And here, buffer size will be the (Pixel) * (size of "window")
    VkBuffer storageBuffer;
    VkDeviceMemory storageBufferMemory;
    VkDeviceSize bufferSize = sizeof(Pixel) * WINDOW_WIDTH * WINDOW_HEIGHT;
    
    // These are the storage buffers we will use
    // for spheres (we are rendering 3 spheres)
    VkBuffer spheresStorageBuffer;
    VkDeviceMemory spheresStorageBufferMemory;
    VkDeviceSize spheresStoragebufferSize = sizeof(Sphere) * 3;
    
    // The pipeline is describes all the commands passes through in Vulkan
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;
    
    // CommanBuffers are used to store record commands submitted to a queue
    // CommanPool is used to allocate the commands
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    
    void initVulkan() {
        createInstance();
        setupDebugCallback();
        
        pickPhysicalDevice();
        createLogicalDevice();
        
        createImageStorageBuffer();
        createObjectsStorageBuffer();
        
        createAndMapObjectsStorageBufferData();
        
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSet();
        
        createComputePipeline();
        
        createCommandPool();
        createCommandBuffer();
        
        runCommandBuffer();
        saveRenderedImage();
    }
    
    void cleanup() {
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkFreeMemory(device, storageBufferMemory, nullptr);
        
        vkDestroyBuffer(device, spheresStorageBuffer, nullptr);
        vkFreeMemory(device, spheresStorageBufferMemory, nullptr);
        
        vkDestroyCommandPool(device, commandPool, nullptr);
        
        vkDestroyDevice(device, nullptr);
        
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
        }
        
        vkDestroyInstance(instance, nullptr);
    }
    
    void createInstance() {
        if(enableValidationLayers && !isValidationLayerSupportAvailable()) {
            throw std::runtime_error("Requested validation layer not available!");
        }
        
        VkApplicationInfo appInfo = {};
        
        // Specify application info
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine Yet";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        // Specifying parameters of a newly created instance
        // Actually uses the previously defined VkApplicationInfo
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
        
        // Actually create the instance
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance!");
        }
        
        // Optional
        checkAvailableExtensions();
    }
    
    void checkAvailableExtensions() {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        
        printf("Available extensions:\n");
        for (const auto& extension : extensions) {
            printf("\t%s\n", extension.extensionName);
        }
    }
    
    bool isValidationLayerSupportAvailable() {
        uint32_t layersCount = 0;
        vkEnumerateInstanceLayerProperties(&layersCount, nullptr);
        
        std::vector<VkLayerProperties> availableLayers(layersCount);
        vkEnumerateInstanceLayerProperties(&layersCount, availableLayers.data());
        
        printf("Available Layers:\n");
        for(const auto& availableLayer : availableLayers) {
            printf("\t%s\n", availableLayer.layerName);
        }
        
        std::set<std::string> requiredLayers(validationLayers.begin(), validationLayers.end());
        
        for (const auto& availableLayer : availableLayers) {
            requiredLayers.erase(availableLayer.layerName);
        }
        
        return requiredLayers.empty();
    }
    
    std::vector<const char*> getRequiredExtensions() {
        std::vector<const char*> extensions;
        
        if(enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        
        return extensions;
    }
    
    void setupDebugCallback() {
        if (!enableValidationLayers) {
            return;
        }
        
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.flags = 0;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug callback!");
        }
    }
    
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    
    void pickPhysicalDevice() {
        // Let's find a device on the PC that cam run Vulkan
        
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support! :(");
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }
    
    bool isDeviceSuitable(VkPhysicalDevice device) {
        printf("Checking devices...\n");
        
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        
        // For now the following is not necessary, we are not checking
        // for specific features
        // VkPhysicalDeviceFeatures deviceFeatures;
        // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        
        printf("Working with device: %s\n", deviceProperties.deviceName);
        
        // Required queue families
        QueueFamilyIndices indices = findQueueFamilies(device);
        
        // For now we will use any device with Vulkan support!
        return indices.isComplete();
    }
    
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        // Check for a queue family with compute commands support (VK_QUEUE_COMPUTE_BIT)
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // compute
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                indices.computeFamily = i;
            }
            
            if (indices.isComplete()) {
                break;
            }
            
            i++;
        }
        
        return indices;
    }
    
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.computeFamily};
        
        // They might be on the same queue family, but this is safe in that case aswell
        float queuePriority = 1.0f;
        for (int queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        // For now, empty
        VkPhysicalDeviceFeatures deviceFeatures = {};
        
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
        
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }
        
        // Get the queues from the devices
        vkGetDeviceQueue(device, indices.computeFamily, 0, &computeQueue);
    }
    
    void createComputePipeline() {
        // Here we will create a pipeline that only uses compute shaders
        // much more simple than graphics pipelines
        
        // Read out shader and create a shader module
        auto computeShaderCode = readFile("shaders/comp.spv");
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
        
        VkPipelineShaderStageCreateInfo computeShaderStageCreateInfo = {};
        computeShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageCreateInfo.module = computeShaderModule;
        computeShaderStageCreateInfo.pName = "main";
        
        // The pipeline layout is set up so the pipeline can access
        // the descriptor sets (so we can actually access our buffers)
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }
        
        // Let's create the compute pipeline!
        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = computeShaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }
    
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }
        
        return shaderModule;
    }

    void createCommandPool() {
        // Before creating the command buffers to send command to the GPU we need
        // to setup a command pool
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // all command buffers set to this command pool must submit commands to this
        // queue family only
        poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily;
        
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }
    
    
    void createCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }
        
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // only submitted once
        
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
        
        // Bind pipeline and and descriptor set
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        
        // This starts the compute pipeline and executes the compute shader
        vkCmdDispatch(commandBuffer, (uint32_t)ceil(WINDOW_WIDTH / float(WORKGROUP_SIZE)),
                      (uint32_t)ceil(WINDOW_HEIGHT / float(WORKGROUP_SIZE)), 1);
        
        // End recording commands
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }
    
    void runCommandBuffer() {
        // We have recorded our command, that essentially just starts the compute
        // pipeline and shader, and now we will actually run the command
        
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        // A fence is used to wait for the command to finish running
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        
        if (vkCreateFence(device, &fenceCreateInfo, NULL, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence!");
        }

        // Now we submit the command buffer on the queue, with the fence
        if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit the command buffer and fence!");
        }
        
        // Since we will read from the buffer the result from the compute shader,
        // we need to wait on the fence, which signilizes the end of the command
        // (with a default timeout)
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS) {
            throw std::runtime_error("Failed wait fot the command to run!");
        }
        
        vkDestroyFence(device, fence, NULL);
    }
    
    void createImageStorageBuffer() {
        // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT -> can use vkMapMemory to read the
        // buffer memory from the GPU to the CPU
        // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT -> can read from the GPU to the CPU
        // without extra steps
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, // used simply as a stored buffer
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     storageBuffer,
                     storageBufferMemory);
    }
    
    void createObjectsStorageBuffer() {
        // First setting up the spheres
        createBuffer(spheresStoragebufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     spheresStorageBuffer,
                     spheresStorageBufferMemory);
    }
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer!");
        }
        
        // Next, allocate memory
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        // If we have found the memory we need on
        // the device, allocate buffer memory in it
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory!");
        }
        
        // Assign the buffer memory to the actual buffer
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    void createAndMapObjectsStorageBufferData() {
        Sphere sphere1 = {};
        sphere1.position = glm::vec3(1.0f, 0.6f, 4.0f);
        sphere1.radius = 0.6f;
        sphere1.albedo = glm::vec3(0.8f, 0.5f, 0.3f);
        sphere1.specular = glm::vec3(0.3f, 0.3f, 0.3f);
        
        Sphere sphere2 = {};
        sphere2.position = glm::vec3(-1.0f, 0.6f, 4.0f);
        sphere2.radius = 0.6f;
        sphere2.albedo = glm::vec3(0.5f, 0.8f, 0.3f);
        sphere2.specular = glm::vec3(0.3f, 0.3f, 0.3f);
        
        Sphere sphere3 = {};
        sphere3.position = glm::vec3(0.0f, 0.8f, 5.5f);
        sphere3.radius = 0.8f;
        sphere3.albedo = glm::vec3(0.5f, 0.3f, 0.8f);
        sphere3.specular = glm::vec3(0.3f, 0.3f, 0.3f);
        
        Sphere spheres[3] = { sphere1, sphere2, sphere3 };
        
        void* data;
        if (vkMapMemory(device, spheresStorageBufferMemory, 0, sizeof(spheres), 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("Error mapping memory!");
        }
        memcpy(data, spheres, sizeof(spheres));
        vkUnmapMemory(device, spheresStorageBufferMemory);
        
        //memcpy(data, &sphere1UBO, sizeof(SphereUBO));
        
        //vkMapMemory(device, spheresUniformBufferMemory, sizeof(SphereUBO), sizeof(SphereUBO), 0, &data);
        //memcpy(data, &sphere2UBO, sizeof(SphereUBO));
        
        //vkMapMemory(device, spheresUniformBufferMemory, 2 * sizeof(SphereUBO), sizeof(SphereUBO), 0, &data);
        //memcpy(data, &sphere3UBO, sizeof(SphereUBO));
        
    }
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    
    void createDescriptorSetLayout() {
        // The descriptor layout specifies the types of resources that are going
        // to be accessed by the pipeline
        // It's what allows us to bind descriptors to the shaders
        //
        // Notice "layoutBinding.binding = 0;", this means on the shader we will
        // access it using
        //
        // layout(binding = 0) buffer storageBuffer
        // and so on for other bindings
        
        // Our image
        VkDescriptorSetLayoutBinding layoutBinding = {};
        layoutBinding.binding = 0;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // Our spheres
        VkDescriptorSetLayoutBinding spheresStorageLayoutBinding = {};
        spheresStorageLayoutBinding.binding = 1;
        spheresStorageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        spheresStorageLayoutBinding.descriptorCount = 1;
        spheresStorageLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
            layoutBinding,
            spheresStorageLayoutBinding
        };
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }
    
    void createDescriptorPool() {
        // For this program, our descriptor pool will allocate
        // storage buffers and uniform buffers
        
        VkDescriptorPoolSize poolSizeStorageBuffer = {};
        poolSizeStorageBuffer.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizeStorageBuffer.descriptorCount = 1; // only one storage buffer
        
        VkDescriptorPoolSize poolSizeUniformBuffer = {};
        poolSizeUniformBuffer.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizeUniformBuffer.descriptorCount = 1; // only one unigorm buffer
        
        VkDescriptorPoolSize poolSizeArray[2] = { poolSizeStorageBuffer, poolSizeUniformBuffer };
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizeArray;
        poolInfo.maxSets = 1;
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }
    
    void createDescriptorSet() {
        // First we created the descriptor pool,
        // now we can create the descriptor set we need.
        
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool; // we will allocate this set from the pool
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set!");
        }
        
        // Descriptor sets that refer to buffers need VkDescriptorBufferInfo to configure it
        VkDescriptorBufferInfo storageBufferInfo = {};
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = bufferSize;
        
        VkWriteDescriptorSet descriptorWriteStorageBuffer = {};
        descriptorWriteStorageBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteStorageBuffer.dstSet = descriptorSet;
        descriptorWriteStorageBuffer.dstBinding = 0;
        descriptorWriteStorageBuffer.dstArrayElement = 0;
        descriptorWriteStorageBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWriteStorageBuffer.descriptorCount = 1;
        descriptorWriteStorageBuffer.pBufferInfo = &storageBufferInfo;
        
        // For our spheres
        VkDescriptorBufferInfo spheresStorageBufferInfo = {};
        spheresStorageBufferInfo.buffer = spheresStorageBuffer;
        spheresStorageBufferInfo.offset = 0;
        spheresStorageBufferInfo.range = spheresStoragebufferSize;
        
        VkWriteDescriptorSet descriptorWriteSpheresStorageBuffer = {};
        descriptorWriteSpheresStorageBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteSpheresStorageBuffer.dstSet = descriptorSet;
        descriptorWriteSpheresStorageBuffer.dstBinding = 1;
        descriptorWriteSpheresStorageBuffer.dstArrayElement = 0;
        descriptorWriteSpheresStorageBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWriteSpheresStorageBuffer.descriptorCount = 1;
        descriptorWriteSpheresStorageBuffer.pBufferInfo = &spheresStorageBufferInfo;
        
        VkWriteDescriptorSet writeDescSets[2] = {
            descriptorWriteStorageBuffer,
            descriptorWriteSpheresStorageBuffer
        };
        vkUpdateDescriptorSets(device, 2, writeDescSets, 0, nullptr);
    }
    
    void saveRenderedImage() {
        void* mappedMemory = nullptr;
        
        // Map the buffer memory, so that we can read it
        vkMapMemory(device, storageBufferMemory, 0, bufferSize, 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *) mappedMemory;
        
        // Get the color data from the buffer, and cast it to bytes.
        std::vector<unsigned char> image;
        image.reserve(WINDOW_WIDTH * WINDOW_HEIGHT * 4);
        for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; ++i) {
            image.push_back((unsigned char)(255.0f * clamp(pmappedMemory[i].r, 0, 1)));
            image.push_back((unsigned char)(255.0f * clamp(pmappedMemory[i].g, 0, 1)));
            image.push_back((unsigned char)(255.0f * clamp(pmappedMemory[i].b, 0, 1)));
            image.push_back((unsigned char)(255.0f * clamp(pmappedMemory[i].a, 0, 1)));
        }
        
        vkUnmapMemory(device, storageBufferMemory);
        
        //Encode the image
        printf("Saving result to image...\n");
        unsigned error = lodepng::encode("result.png", image, WINDOW_WIDTH, WINDOW_HEIGHT);
        
        //if there's an error, display it
        if (error) {
            printf("Encoder error %d: %s\n", error, lodepng_error_text(error));
            throw std::runtime_error("Error enconding png image!");
        }
    }
    
    double clamp(double x, double lower, double upper) {
        return std::min(upper, std::max(x, lower));
    }
};

int main() {
    RenderTriangle app;
    
    try {
        app.run();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
