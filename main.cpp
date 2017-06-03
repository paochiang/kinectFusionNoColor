#include<string>
#include<windows.h>
#include<iostream>
#include<fstream>
#include<Kinect.h>
#include <NuiKinectFusionApi.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef SAFE_FUSION_RELEASE_IMAGE_FRAME
#define SAFE_FUSION_RELEASE_IMAGE_FRAME(p) { if (p) { static_cast<void>(NuiFusionReleaseImageFrame(p)); (p)=NULL; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p); (p)=NULL; } }
#endif

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}
/// Set Identity in a Matrix4
void SetIdentityMatrix(Matrix4 &mat)
{
	mat.M11 = 1; mat.M12 = 0; mat.M13 = 0; mat.M14 = 0;
	mat.M21 = 0; mat.M22 = 1; mat.M23 = 0; mat.M24 = 0;
	mat.M31 = 0; mat.M32 = 0; mat.M33 = 1; mat.M34 = 0;
	mat.M41 = 0; mat.M42 = 0; mat.M43 = 0; mat.M44 = 1;
}

void UpdateIntrinsics(NUI_FUSION_IMAGE_FRAME * pImageFrame, NUI_FUSION_CAMERA_PARAMETERS * params)
{
	if (pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr)
	{
		pImageFrame->pCameraParameters->focalLengthX = params->focalLengthX;
		pImageFrame->pCameraParameters->focalLengthY = params->focalLengthY;
		pImageFrame->pCameraParameters->principalPointX = params->principalPointX;
		pImageFrame->pCameraParameters->principalPointY = params->principalPointY;
	}

	// Confirm we are called correctly
	_ASSERT(pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr);
}


class KinectFusion
{
public:
	KinectFusion();
	HRESULT CreateFirstConnected();
	HRESULT InitializeKinectFusion();
	HRESULT ResetReconstruction();
	HRESULT SetupUndistortion();
	HRESULT OnCoordinateMappingChanged();
	void closeSensor() {
		if (m_pNuiSensor) {
			m_pNuiSensor->Close();
		}
	};
	bool coordinateChange();
	void increaseFusion(Mat curDepth);
	void mProcessDepth();
	void processMesh();
	void processDepthToCloud();
	~KinectFusion();
	int                         m_cFrameCounter;				//camera tracking中已经有的frame的个数
	bool						m_bTrackingFailed;		//camera tracking是否失败标志
	int							m_cLostFrameCounter;		//连续帧之间丢失的帧数
private:

	IMultiSourceFrameReader* m_pMultiFrameReader;

	// Current Kinect
	IKinectSensor*              m_pNuiSensor;

	/// For depth distortion correction
	ICoordinateMapper*          m_pMapper;									//坐标系转换
	WAITABLE_HANDLE             m_coordinateMappingChangedEvent;			//坐标系映射变换标志

	static const UINT                         m_cDepthWidth = 512;
	static const UINT                         m_cDepthHeight = 424;
	static const UINT                         m_cDepthImagePixels = 512 * 424;

	static const int            cBytesPerPixel = 4; // for depth float and int-per-pixel raycast images
	CameraSpacePoint* m_pCameraCoordinates;	//3D空间点坐标（相机坐标系框架）
	Mat i_depth;	//用于显示的深度图，取深度图16位的后8位
	UINT16* depthData;	//深度数据
	NUI_FUSION_RECONSTRUCTION_PARAMETERS m_reconstructionParams;	//volume 的参数
	float                       m_fMinDepthThreshold;		//对获得的depth图，通过这两个阈值进行第一步过滤，单位为米
	float                       m_fMaxDepthThreshold;
	unsigned short              m_cMaxIntegrationWeight;	//将深度图整合进global model时用到的临时平均参数,越小噪声点越多，适合动态；越大融合越慢，细节更多，噪声点更少。
	int                         m_deviceIndex;						//当使用GPU时，选择的设备的索引
	NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE m_processorType;		//使用GPU或CPU

	/// The Kinect Fusion Reconstruction Volume
	INuiFusionReconstruction*   m_pVolume;						//Volume
	// The Kinect Fusion Camera Transform
	Matrix4                     m_worldToCameraTransform;		//global到camera坐标系的转换

	// The default Kinect Fusion World to Volume Transform
	Matrix4                     m_defaultWorldToVolumeTransform; //默认的global到camera坐标系的转换矩阵

	NUI_FUSION_CAMERA_PARAMETERS m_cameraParameters;		//camera参数 focalx,focaly, principalPointX, principalPointsY;

	/// Frames from the depth input
	UINT16*                     m_pDepthImagePixelBuffer;	//经过原始的深度图经过m_pDepthDistortionLT过滤后的深度数据
	NUI_FUSION_IMAGE_FRAME*     m_pDepthFloatImage;			//将m_pDepthImagePixelBuffer用m_fMinDepthThreshold和m_fMaxDepthThreshold过滤后的深度数据。
	NUI_FUSION_IMAGE_FRAME*		m_pSmoothDepthFloatImage;	//将m_pDepthFloatImage平滑后的深度数据帧
	NUI_FUSION_IMAGE_FRAME*     m_pPointCloud;	//第i帧光线投影结果，(作为本帧的深度结果)用于和下一帧进行tracking

	/// Images for display
	NUI_FUSION_IMAGE_FRAME*     m_pShadedSurface;			//三角化后的模型
	DepthSpacePoint*            m_pDepthDistortionMap;	//深度图坐标（每个值代表深度图上的坐标）
	UINT*                       m_pDepthDistortionLT;

	bool						m_bHaveValidCameraParameters;		//是否有合法的相机参数
	bool                        m_bInitializeError;		//初始化是否成功标志
	bool                        m_bMirrorDepthFrame;
	bool						m_bTranslateResetPoseByMinDepthThreshold;	//是否将volume往z轴正方向（world frame坐标系）平移

};

KinectFusion::KinectFusion() {
	m_pMultiFrameReader = nullptr;
	i_depth.create(m_cDepthHeight, m_cDepthWidth, CV_8UC1);
	m_pNuiSensor = nullptr;
	m_pMapper = nullptr;
	m_coordinateMappingChangedEvent = NULL;
	depthData = new UINT16[m_cDepthHeight * m_cDepthWidth];
	m_pCameraCoordinates = new CameraSpacePoint[m_cDepthImagePixels];
	// Define a cubic Kinect Fusion reconstruction volume,
	// with the Kinect at the center of the front face and the volume directly in front of Kinect.
	m_reconstructionParams.voxelsPerMeter = 256;// 1000mm / 256vpm = ~3.9mm/voxel    
	m_reconstructionParams.voxelCountX = 384;   // 384 / 256vpm = 1.5m wide reconstruction
	m_reconstructionParams.voxelCountY = 384;   // Memory = 384*384*384 * 4bytes per voxel
	m_reconstructionParams.voxelCountZ = 384;   // This will require a GPU with at least 256MB

	m_fMinDepthThreshold = NUI_FUSION_DEFAULT_MINIMUM_DEPTH;   // min depth in meters
	m_fMaxDepthThreshold = NUI_FUSION_DEFAULT_MAXIMUM_DEPTH;    // max depth in meters
	// This parameter is the temporal averaging parameter for depth integration into the reconstruction
	m_cMaxIntegrationWeight = NUI_FUSION_DEFAULT_INTEGRATION_WEIGHT;	// Reasonable for static scenes
	m_deviceIndex = -1;	//自动选择GPU设备
	m_processorType = NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_AMP; //使用GPU或CPU

	SetIdentityMatrix(m_worldToCameraTransform);
	SetIdentityMatrix(m_defaultWorldToVolumeTransform);

	// We don't know these at object creation time, so we use nominal values.
	// These will later be updated in response to the CoordinateMappingChanged event.
	m_cameraParameters.focalLengthX = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_X;		//这个值在开始是不知道的，但在后面会的得到并需要更新
	m_cameraParameters.focalLengthY = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_Y;
	m_cameraParameters.principalPointX = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_X;
	m_cameraParameters.principalPointY = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_Y;
	m_pVolume = NULL;
	m_pDepthFloatImage = nullptr;
	m_pSmoothDepthFloatImage = nullptr;
	m_pDepthImagePixelBuffer = nullptr;
	m_pPointCloud = nullptr;
	m_pShadedSurface = nullptr;
	m_pDepthDistortionMap = nullptr;
	m_pDepthDistortionLT = nullptr;
	m_bHaveValidCameraParameters = false;
	m_bInitializeError = false;
	m_bMirrorDepthFrame = false;
	m_bTrackingFailed = false;
	m_cFrameCounter = 0;
	m_cLostFrameCounter = 0;
	m_bTranslateResetPoseByMinDepthThreshold = true;

}
KinectFusion::~KinectFusion() {

	if (m_pCameraCoordinates) {
		delete[] m_pCameraCoordinates;
		m_pCameraCoordinates = NULL;
	}
	SafeRelease(m_pMultiFrameReader);
	SafeRelease(m_pMapper);
	if (nullptr != m_pMapper)
		m_pMapper->UnsubscribeCoordinateMappingChanged(m_coordinateMappingChangedEvent);
	if (m_pNuiSensor) {
		m_pNuiSensor->Close();
	}
	SafeRelease(m_pNuiSensor);
	SafeRelease(m_pVolume);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pDepthFloatImage);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pSmoothDepthFloatImage);
	SAFE_DELETE_ARRAY(m_pDepthImagePixelBuffer);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pPointCloud);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pShadedSurface);
	SAFE_DELETE_ARRAY(m_pDepthDistortionMap);
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	SAFE_DELETE_ARRAY(depthData)

}
HRESULT KinectFusion::CreateFirstConnected()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pNuiSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pNuiSensor) {
		hr = m_pNuiSensor->Open();
		if (SUCCEEDED(hr))
			hr = m_pNuiSensor->get_CoordinateMapper(&m_pMapper);
		if (SUCCEEDED(hr))
			hr = m_pMapper->SubscribeCoordinateMappingChanged(&m_coordinateMappingChangedEvent);
	}
	if (nullptr == m_pNuiSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!" << endl;
		return E_FAIL;
	}
	return hr;
}
HRESULT KinectFusion::InitializeKinectFusion() {
	HRESULT hr = S_OK;

	//设备检查
	// Check to ensure suitable DirectX11 compatible hardware exists before initializing Kinect Fusion
	WCHAR description[MAX_PATH];    //The description of the device.
	WCHAR instancePath[MAX_PATH];	//The DirectX instance path of the GPU being used for reconstruction.
	UINT memorySize = 0;
	if (FAILED(hr = NuiFusionGetDeviceInfo(m_processorType, m_deviceIndex, &description[0], ARRAYSIZE(description), &instancePath[0], ARRAYSIZE(instancePath), &memorySize)))
	{
		if (hr == E_NUI_BADINDEX)
		{
			// This error code is returned either when the device index is out of range for the processor 
			// type or there is no DirectX11 capable device installed. As we set -1 (auto-select default) 
			// for the device index in the parameters, this indicates that there is no DirectX11 capable 
			// device. The options for users in this case are to either install a DirectX11 capable device
			// (see documentation for recommended GPUs) or to switch to non-real-time CPU based 
			// reconstruction by changing the processor type to NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_CPU.
			cout << "No DirectX11 device detected, or invalid device index - Kinect Fusion requires a DirectX11 device for GPU-based reconstruction." << endl;
		}
		else
		{
			cout << "Failed in call to NuiFusionGetDeviceInfo." << endl;
		}
		return hr;
	}
	//创建Fusion 容积重建 Volume
	hr = NuiFusionCreateReconstruction(&m_reconstructionParams, m_processorType, m_deviceIndex, &m_worldToCameraTransform, &m_pVolume);
	if (FAILED(hr))
	{
		if (E_NUI_GPU_FAIL == hr)
		{
			cout << "Device " << m_deviceIndex << " not able to run Kinect Fusion, or error initializing." << endl;
		}
		else if (E_NUI_GPU_OUTOFMEMORY == hr)
		{

			cout << "Device " << m_deviceIndex << " out of memory error initializing reconstruction - try a smaller reconstruction volume." << endl;

		}
		else if (NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_CPU != m_processorType)
		{
			cout << "Failed to initialize Kinect Fusion reconstruction volume on device" << m_deviceIndex << endl;
		}
		else
		{
			cout << "Failed to initialize Kinect Fusion reconstruction volume on CPU." << endl;
		}
		return hr;
	}

	//保存创建的volume的WorldToVolumeTransform矩阵
	// Save the default world to volume transformation to be optionally used in ResetReconstruction
	hr = m_pVolume->GetCurrentWorldToVolumeTransform(&m_defaultWorldToVolumeTransform);
	if (FAILED(hr))
	{
		cout << "Failed in call to GetCurrentWorldToVolumeTransform." << endl;
		return hr;
	}
	//是否将volume在z轴（基于world frame坐标系）往正方向平移一段voxels(因为kinect最小的depth值也会大于0.5m左右，故就算在volume里重建这一部分，也没有模型)
	if (m_bTranslateResetPoseByMinDepthThreshold)
	{
		//重新构建volume
		hr = ResetReconstruction();
		if (FAILED(hr))
		{
			return hr;
		}
	}

	//创建浮点深度帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_FLOAT, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pDepthFloatImage);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pDepthFloatImage." << endl;
		return hr;
	}
	// 创建平滑浮点深度帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_FLOAT, m_cDepthWidth, m_cDepthHeight, nullptr, &m_pSmoothDepthFloatImage);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pSmoothDepthFloatImage." << endl;
		return hr;
	}

	// 创建光线投影点云帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_POINT_CLOUD, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pPointCloud);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pPointCloud." << endl;
		return hr;
	}
	// 表面点云着色帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pShadedSurface);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pShadedSurface." << endl;
		return hr;
	}

	//分配深度图相关的内存
	_ASSERT(m_pDepthImagePixelBuffer == nullptr);
	m_pDepthImagePixelBuffer = new(std::nothrow) UINT16[m_cDepthImagePixels];
	if (nullptr == m_pDepthImagePixelBuffer)
	{
		cout << "Failed to initialize Kinect Fusion depth image pixel buffer." << endl;
		return hr;
	}
	_ASSERT(m_pDepthDistortionMap == nullptr);
	m_pDepthDistortionMap = new(std::nothrow) DepthSpacePoint[m_cDepthImagePixels];
	if (nullptr == m_pDepthDistortionMap)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion buffer." << endl;
		return E_OUTOFMEMORY;
	}
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	m_pDepthDistortionLT = new(std::nothrow) UINT[m_cDepthImagePixels];

	if (nullptr == m_pDepthDistortionLT)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion Lookup Table." << endl;
		return E_OUTOFMEMORY;
	}
	// If we have valid parameters, let's go ahead and use them.
	if (m_cameraParameters.focalLengthX != 0)
	{
		SetupUndistortion();
	}
	return hr;
}
HRESULT KinectFusion::ResetReconstruction()
{
	if (nullptr == m_pVolume)
	{
		return E_FAIL;
	}

	HRESULT hr = S_OK;

	SetIdentityMatrix(m_worldToCameraTransform);

	//将volume在Z轴上往正方向平移最小深度大小的voxels，这样对于最小深度的world points，它在volume里的z轴对应为0.（volume可以理解一个操作区域，只对此区域内的点云进行处理）
	if (m_bTranslateResetPoseByMinDepthThreshold)
	{
		Matrix4 worldToVolumeTransform = m_defaultWorldToVolumeTransform;

		// 将volume在Z轴上平移最小深度大小的voxels(从0到最小深度内不会有重建的模型，故不需要再这些部分创建volume)
		float minDist = (m_fMinDepthThreshold < m_fMaxDepthThreshold) ? m_fMinDepthThreshold : m_fMaxDepthThreshold;
		worldToVolumeTransform.M43 -= (minDist * m_reconstructionParams.voxelsPerMeter);

		//worldToVolumeTransform.M42 += (0.2553 * m_reconstructionParams.voxelsPerMeter);
		//worldToVolumeTransform.M42 += (0.33 * m_reconstructionParams.voxelsPerMeter);
		//worldToVolumeTransform.M42 += (0.24 * m_reconstructionParams.voxelsPerMeter);
		//worldToVolumeTransform.M42 += (0.28 * m_reconstructionParams.voxelsPerMeter); //花盆
		//worldToVolumeTransform.M42 += (0.1* m_reconstructionParams.voxelsPerMeter);

		hr = m_pVolume->ResetReconstruction(&m_worldToCameraTransform, &worldToVolumeTransform);
	}
	else
	{
		hr = m_pVolume->ResetReconstruction(&m_worldToCameraTransform, nullptr);
	}

	m_cLostFrameCounter = 0;
	m_cFrameCounter = 0;

	if (SUCCEEDED(hr))
	{
		m_bTrackingFailed = false;
		cout << "Reconstruction has been reset." << endl;
	}
	else
	{
		cout << "Failed to reset reconstruction." << endl;
	}

	return hr;
}
HRESULT KinectFusion::SetupUndistortion()
{
	HRESULT hr = E_UNEXPECTED;

	//深度图坐标系原点不能在图像中心，否则此摄像机参数就不合法
	if (m_cameraParameters.principalPointX != 0)
	{
		//深度图的四个边坐标：左上（0，0），右上（1，0）（因为k矩阵里参数分别都除以深度图的宽和高），左下（0，1），右下（1，1）反投影成camera frame 下且z为1（1m）的空间点
		CameraSpacePoint cameraFrameCorners[4] = //at 1 meter distance. Take into account that depth frame is mirrored
		{
			/*LT*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*RT*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*LB*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f },
			/*RB*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f }
		};

		//将4个1m处的空间点边界内的空间划分为个数和深度图大小相同的空间点，然后将这些点投影回深度图上。
		for (UINT rowID = 0; rowID < m_cDepthHeight; rowID++)
		{
			const float rowFactor = float(rowID) / float(m_cDepthHeight - 1);
			const CameraSpacePoint rowStart =
			{
				cameraFrameCorners[0].X + (cameraFrameCorners[2].X - cameraFrameCorners[0].X) * rowFactor,
				cameraFrameCorners[0].Y + (cameraFrameCorners[2].Y - cameraFrameCorners[0].Y) * rowFactor,
				1.f
			};

			const CameraSpacePoint rowEnd =
			{
				cameraFrameCorners[1].X + (cameraFrameCorners[3].X - cameraFrameCorners[1].X) * rowFactor,
				cameraFrameCorners[1].Y + (cameraFrameCorners[3].Y - cameraFrameCorners[1].Y) * rowFactor,
				1.f
			};

			const float stepFactor = 1.f / float(m_cDepthWidth - 1);
			const CameraSpacePoint rowDelta =
			{
				(rowEnd.X - rowStart.X) * stepFactor,
				(rowEnd.Y - rowStart.Y) * stepFactor,
				0
			};

			_ASSERT(m_cDepthWidth == NUI_DEPTH_RAW_WIDTH);
			CameraSpacePoint cameraCoordsRow[NUI_DEPTH_RAW_WIDTH];

			CameraSpacePoint currentPoint = rowStart;
			for (UINT i = 0; i < m_cDepthWidth; i++)
			{
				cameraCoordsRow[i] = currentPoint;
				currentPoint.X += rowDelta.X;
				currentPoint.Y += rowDelta.Y;
			}

			hr = m_pMapper->MapCameraPointsToDepthSpace(m_cDepthWidth, cameraCoordsRow, m_cDepthWidth, &m_pDepthDistortionMap[rowID * m_cDepthWidth]);
			if (FAILED(hr))
			{
				cout << "Failed to initialize Kinect Coordinate Mapper." << endl;
				return hr;
			}
		}

		if (nullptr == m_pDepthDistortionLT)
		{
			cout << "Failed to initialize Kinect Fusion depth image distortion Lookup Table." << endl;
			return E_OUTOFMEMORY;
		}

		//若反投影回的深度图位置不合法，这将此处位置的深度图标记为不可从空间坐标投影回来，用于后面过滤采集到的深度图
		UINT* pLT = m_pDepthDistortionLT;
		for (UINT i = 0; i < m_cDepthImagePixels; i++, pLT++)
		{
			//nearest neighbor depth lookup table 
			UINT x = UINT(m_pDepthDistortionMap[i].X + 0.5f);
			UINT y = UINT(m_pDepthDistortionMap[i].Y + 0.5f);

			*pLT = (x < m_cDepthWidth && y < m_cDepthHeight) ? x + y * m_cDepthWidth : UINT_MAX;
		}
		m_bHaveValidCameraParameters = true;
	}
	else
	{
		m_bHaveValidCameraParameters = false;
	}
	return S_OK;
}
HRESULT KinectFusion::OnCoordinateMappingChanged()
{
	HRESULT hr = E_UNEXPECTED;

	// Calculate the down sampled image sizes, which are used for the AlignPointClouds calculation frames
	CameraIntrinsics intrinsics = {};

	m_pMapper->GetDepthCameraIntrinsics(&intrinsics);

	float focalLengthX = intrinsics.FocalLengthX / NUI_DEPTH_RAW_WIDTH;
	float focalLengthY = intrinsics.FocalLengthY / NUI_DEPTH_RAW_HEIGHT;
	float principalPointX = intrinsics.PrincipalPointX / NUI_DEPTH_RAW_WIDTH;
	float principalPointY = intrinsics.PrincipalPointY / NUI_DEPTH_RAW_HEIGHT;

	if (m_cameraParameters.focalLengthX == focalLengthX && m_cameraParameters.focalLengthY == focalLengthY &&
		m_cameraParameters.principalPointX == principalPointX && m_cameraParameters.principalPointY == principalPointY)
		return S_OK;

	m_cameraParameters.focalLengthX = focalLengthX;
	m_cameraParameters.focalLengthY = focalLengthY;
	m_cameraParameters.principalPointX = principalPointX;
	m_cameraParameters.principalPointY = principalPointY;

	_ASSERT(m_cameraParameters.focalLengthX != 0);

	UpdateIntrinsics(m_pDepthFloatImage, &m_cameraParameters);
	UpdateIntrinsics(m_pPointCloud, &m_cameraParameters);
	UpdateIntrinsics(m_pShadedSurface, &m_cameraParameters);
	UpdateIntrinsics(m_pSmoothDepthFloatImage, &m_cameraParameters);

	if (nullptr == m_pDepthDistortionMap)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion buffer." << endl;
		return E_OUTOFMEMORY;
	}

	hr = SetupUndistortion();
	return hr;
}

bool KinectFusion::coordinateChange() {
	if (nullptr == m_pNuiSensor)
	{
		cout << "cannot get kinect sensor!" << endl;

		exit(0);
	}
	//检查相机参数变化
	if (m_coordinateMappingChangedEvent != NULL && WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)m_coordinateMappingChangedEvent, 0))
	{
		cout << "camere corrdinate map chainge!" << endl;
		OnCoordinateMappingChanged();
		ResetEvent((HANDLE)m_coordinateMappingChangedEvent);
		return true;
	}
	return false;
}
void KinectFusion::increaseFusion(Mat curDepth) {

	for (int row = 0; row < m_cDepthHeight; row++) {
		for (int col = 0; col < m_cDepthWidth; col++) {
			depthData [row * m_cDepthWidth + col]= curDepth.at<unsigned short>(row, col);
		}
	}

	//copy and remap depth
	const UINT bufferLength = m_cDepthImagePixels;
	UINT16 * pDepth = m_pDepthImagePixelBuffer;
	for (UINT i = 0; i < bufferLength; i++, pDepth++)
	{
		const UINT id = m_pDepthDistortionLT[i];
		*pDepth = id < bufferLength ? depthData[id] : 0;
	}
	mProcessDepth();


}
void KinectFusion::processDepthToCloud() {
	int count = 0;
	for (int i = 0; i < m_cDepthImagePixels; i++)
	{
		CameraSpacePoint p = m_pCameraCoordinates[i];
		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
		{
			count++;
		}
	}
	ofstream ofs(".\\out.ply");
	string num;
	stringstream ss;
	ss << count;
	ss >> num;
	string str = "ply\nformat ascii 1.0\nelement face 0\n property list uchar int vertex_indices\nelement vertex " + string(num) + "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nproperty uchar alpha\nend_header\n";
	ofs << str;
	for (int i = 0; i < m_cDepthImagePixels; i++)
	{
		CameraSpacePoint p = m_pCameraCoordinates[i];
		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
		{
			float cameraX = static_cast<float>(p.X);
			float cameraY = static_cast<float>(p.Y);
			float cameraZ = static_cast<float>(p.Z);
			ofs << cameraX << ' ' << cameraY << ' ' << cameraZ << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << endl;
		}
	}
	ofs.close();
}
void KinectFusion::mProcessDepth() {
	if (m_bInitializeError)
	{
		cout << "m_bInitializeError ！" << endl;
		return;
	}

	HRESULT hr = S_OK;
	if (nullptr == m_pVolume)
	{
		cout << "Kinect Fusion reconstruction volume not initialized. Please try reducing volume size or restarting." << endl;
		return;
	}
	//由原深度数据构造浮点数据
	hr = m_pVolume->DepthToDepthFloatFrame(m_pDepthImagePixelBuffer, m_cDepthImagePixels * sizeof(UINT16), m_pDepthFloatImage, m_fMinDepthThreshold, m_fMaxDepthThreshold, m_bMirrorDepthFrame);
	//hr = m_pVolume->DepthToDepthFloatFrame(depthData, m_cDepthImagePixels * sizeof(UINT16), m_pDepthFloatImage, m_fMinDepthThreshold, m_fMaxDepthThreshold, true);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion NuiFusionDepthToDepthFloatFrame call failed." << endl;
		return;
	}

	// 平滑深度数据
	hr = m_pVolume->SmoothDepthFloatFrame(m_pDepthFloatImage, m_pSmoothDepthFloatImage, 1, 0.03f);
	if (FAILED(hr)){
		cout << "Kinect Fusion SmoothDepthFloatFrame call failed." << endl;
		return;
	}

	//处理当前帧， 进行 camera tracking 和 update the Kinect Fusion Volume
	// This will create memory on the GPU, upload the image, run camera tracking and integrate the
	// data into the Reconstruction Volume if successful. Note that passing nullptr as the final 
	// parameter will use and update the internal camera pose.
	if (SUCCEEDED(hr))
		hr = m_pVolume->ProcessFrame(m_pSmoothDepthFloatImage, NUI_FUSION_DEFAULT_ALIGN_ITERATION_COUNT, m_cMaxIntegrationWeight, nullptr, &m_worldToCameraTransform);

	// 检查 camera tracking 是否失败. 
	if (FAILED(hr))
	{
		if (hr == E_NUI_FUSION_TRACKING_ERROR)
		{
			m_cLostFrameCounter++;
			m_bTrackingFailed = true;
			cout << "Kinect Fusion camera tracking failed! Align the camera to the last tracked position. " << endl;
		}
		else
		{
			cout << "Kinect Fusion ProcessFrame call failed!" << endl;
			return;
		}
	}
	else
	{
		Matrix4 calculatedCameraPose;
		hr = m_pVolume->GetCurrentWorldToCameraTransform(&calculatedCameraPose);
		if (SUCCEEDED(hr))
		{
			if (m_bTrackingFailed)
				cout << "lostFrameCounter:" << m_cLostFrameCounter << endl;
			// Set the pose
			m_worldToCameraTransform = calculatedCameraPose;
			m_cLostFrameCounter = 0;
			m_bTrackingFailed = false;

		}
	}

	// CalculatePointCloud
	// Raycast all the time, even if we camera tracking failed, to enable us to visualize what is happening with the system
	hr = m_pVolume->CalculatePointCloud(m_pPointCloud, &m_worldToCameraTransform);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion CalculatePointCloud call failed." << endl;
		return;
	}

	// ShadePointCloud and render
	Matrix4 worldToBGRTransform = { 0.0f };
	worldToBGRTransform.M11 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountX;
	worldToBGRTransform.M22 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountY;
	worldToBGRTransform.M33 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountZ;
	worldToBGRTransform.M41 = 0.5f;
	worldToBGRTransform.M42 = 0.5f;
	worldToBGRTransform.M43 = 0.0f;
	worldToBGRTransform.M44 = 1.0f;
	hr = NuiFusionShadePointCloud(m_pPointCloud, &m_worldToCameraTransform, &worldToBGRTransform, m_pShadedSurface, nullptr);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion NuiFusionShadePointCloud call failed." << endl;
		return;
	}
	m_cFrameCounter++;
}
void KinectFusion::processMesh() {
	//mesh
	INuiFusionMesh* infm;
	HRESULT hr = m_pVolume->CalculateMesh(1, &infm);
	if (FAILED(hr)) {
		cout << "CalculateMesh failed!" << endl;
		return;
	}
	unsigned int count = infm->VertexCount();
	unsigned int nCount = infm->NormalCount();
	unsigned int tCount = infm->TriangleVertexIndexCount() / 3;
	cout << "vertex :" << count << endl;
	cout << "normal:" << nCount << endl;
	cout << "triangles:" << tCount << endl;
	const Vector3 *pVertices;// = new Vector3[count];
	hr = infm->GetVertices(&pVertices);
	if (FAILED(hr)) {
		cout << "get Vertices failed!" << endl;
		return;
	}
	const Vector3* pNormals;
	hr = infm->GetNormals(&pNormals);
	if (FAILED(hr)) {
		cout << "get Normals failed!" << endl;
		return;
	}
	ofstream ofs(".\\result.ply");
	string num;
	stringstream ss;
	ss << count;
	ss >> num;
	string str = "ply\nformat ascii 1.0\nelement vertex " + string(num) + "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nproperty uchar alpha\n";
	stringstream ss2;
	string num2;
	ss2 << tCount;
	ss2 >> num2;
	str += "element face " + string(num2) + "\nproperty list uchar int vertex_index\nend_header\n";
	ofs << str;
	for (unsigned i = 0; i < count; i++)
	{
		ofs << pVertices[i].x << ' ' << pVertices[i].y << ' ' << pVertices[i].z << ' ' << pNormals[i].x << ' ' << pNormals[i].y << ' ' << pNormals[i].z << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << endl;
	}
	for (unsigned i = 0; i < nCount; i = i + 3) {
		ofs << static_cast<int>(3) << ' ' << static_cast<int>(i) << ' ' << static_cast<int>(i + 1) << ' ' << static_cast<int>(i + 2) << endl;
	}
	ofs.close();
}

void writeList() {
	ofstream ofs(".\\in.txt");
	//1737
	for (int i = 0; i <= 224; i++) {
		stringstream ss;
		ss << i << endl;
		string out;
		ss >> out;
		while (out.length() < 4) {
			out = "0" + out;
		}
		ofs << "H:/data/fg/fg_depth_" + out + ".png"<< endl;
	}
	ofs.close();
}
int  main() {

	writeList();
	string imageDir = ".\\in.txt";
	//read it to the storage
	vector<string> pics;
	ifstream readPic(imageDir);
	if (!readPic)
	{
		cout<<"error: Cannot open the dir!"<<endl;
		exit(0);
	}
	string tmpStr;
	while (getline(readPic, tmpStr))
	{
		pics.push_back(tmpStr);
	}
	readPic.close();

	KinectFusion kf;
	HRESULT hr = kf.CreateFirstConnected();
	if (FAILED(hr)) {
		cout << "CreateFirstConnected error !" << endl;
		return 0;
	}
	hr = kf.InitializeKinectFusion();
	if (FAILED(hr)) {
		cout << "InitializeKinectFusion error!" << endl;
		return 0;
	}
	while(!kf.coordinateChange());
	kf.closeSensor();
	for (int i = 0; i < pics.size(); i++) {
		cout << "i:" << i << endl;
		Mat m = imread(pics[i], CV_LOAD_IMAGE_ANYDEPTH);
		CV_Assert(!m.empty());
		kf.increaseFusion(m);
		if (kf.m_cLostFrameCounter >= 30) 
			break;
	}
	kf.processMesh();	
	cout << "1114" << endl;
	cout << "帧数：" << kf.m_cFrameCounter << endl;

	return 0;
}
