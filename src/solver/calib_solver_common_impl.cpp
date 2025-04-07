// iKalibr: Unified Targetless Spatiotemporal Calibration Framework
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/iKalibr.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "calib/calib_data_manager.h"
#include "calib/calib_param_manager.h"
#include "calib/ceres_callback.h"
#include "calib/estimator.h"
#include "calib/spat_temp_priori.h"
#include "core/colmap_data_io.h"
#include "core/optical_flow_trace.h"
#include "core/vision_only_sfm.h"
#include "factor/data_correspondence.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "pangolin/display/display.h"
#include "ros/package.h"
#include "sensor/camera_data_loader.h"
#include "solver/calib_solver.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "util/tqdm.h"
#include "util/utils_tpl.hpp"
#include "viewer/viewer.h"
#include "core/haste_data_io.h"
#include "core/event_preprocessing.h"

namespace {
bool IKALIBR_UNIQUE_NAME(_2_) = ns_ikalibr::_1_(__FILE__);
}

namespace ns_ikalibr {

// ----------
// ImagesInfo
// ----------

std::optional<std::string> ImagesInfo::GetImagePath(ns_veta::IndexT id) const {
    auto iter = images.find(id);
    if (iter == images.cend()) {
        return {};
    } else {
        return root_path + '/' + iter->second;
    }
}

std::optional<std::string> ImagesInfo::GetImageFilename(ns_veta::IndexT id) const {
    auto iter = images.find(id);
    if (iter == images.cend()) {
        return {};
    } else {
        return iter->second;
    }
}

std::map<ns_veta::IndexT, std::string> ImagesInfo::GetImagesIdxToName() const { return images; }

std::map<std::string, ns_veta::IndexT> ImagesInfo::GetImagesNameToIdx() const {
    std::map<std::string, ns_veta::IndexT> imgsInvKV;
    for (const auto &[k, v] : images) {
        imgsInvKV.insert({v, k});
    }
    return imgsInvKV;
}

// -----------
// CalibSolver
// -----------

CalibSolver::CalibSolver(CalibDataManager::Ptr calibDataManager,
                         CalibParamManager::Ptr calibParamManager)
    : _dataMagr(std::move(calibDataManager)),
      _parMagr(std::move(calibParamManager)),
      _priori(nullptr),
      _ceresOption(Estimator::DefaultSolverOptions(
          Configor::Preference::AvailableThreads(), true, Configor::Preference::UseCudaInSolving)),
      _viewer(nullptr),
      _initAsset(new InitAsset),
      _solveFinished(false) {
    // create so3 and linear scale splines given start and end times, knot distances
    _splines = CreateSplineBundle(
        _dataMagr->GetCalibStartTimestamp(), _dataMagr->GetCalibEndTimestamp(),
        Configor::Prior::KnotTimeDist::SO3Spline, Configor::Prior::KnotTimeDist::ScaleSpline);

    // create viewer
    _viewer = Viewer::Create(_parMagr, _splines);
    auto modelPath = ros::package::getPath("ikalibr") + "/model/ikalibr.obj";
    _viewer->FillEmptyViews(modelPath);

    // pass the 'CeresViewerCallBack' to ceres option so that update the viewer after every
    // iteration in ceres
    _ceresOption.callbacks.push_back(new CeresViewerCallBack(_viewer));
    _ceresOption.update_state_every_iteration = true;

    // output spatiotemporal parameters after each iteration if needed
    if (IsOptionWith(OutputOption::ParamInEachIter, Configor::Preference::Outputs)) {
        _ceresOption.callbacks.push_back(new CeresDebugCallBack(_parMagr));
    }

    // spatial and temporal priori
    if (std::filesystem::exists(Configor::Prior::SpatTempPrioriPath)) {
        _priori = SpatialTemporalPriori::Load(Configor::Prior::SpatTempPrioriPath);
        _priori->CheckValidityWithConfigor();
        spdlog::info("priori about spatial and temporal parameters are given: '{}'",
                     Configor::Prior::SpatTempPrioriPath);
    }
}

CalibSolver::Ptr CalibSolver::Create(const CalibDataManager::Ptr &calibDataManager,
                                     const CalibParamManager::Ptr &calibParamManager) {
    return std::make_shared<CalibSolver>(calibDataManager, calibParamManager);
}

CalibSolver::~CalibSolver() {
    // solving is not performed or not finished as an exception is thrown
    if (!_solveFinished) {
        pangolin::QuitAll();
    }
    // solving is finished (when use 'pangolin::QuitAll()', the window not quit immediately)
    while (_viewer->IsActive()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurBrToW(double timeByBr) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurBrToW' error, scale spline is not translation spline!!!");
    }
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);
    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        return Sophus::SE3d(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurLkToW(double timeByLk, const std::string &topic) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurLkToW' error, scale spline is not translation spline!!!");
    }
    double timeByBr = timeByLk + _parMagr->TEMPORAL.TO_LkToBr.at(topic);
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);

    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        Sophus::SE3d curBrToW(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
        return curBrToW * _parMagr->EXTRI.SE3_LkToBr(topic);
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurCmToW(double timeByCm, const std::string &topic) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurCmToW' error, scale spline is not translation spline!!!");
    }
    double timeByBr = timeByCm + _parMagr->TEMPORAL.TO_CmToBr.at(topic);
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);

    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        Sophus::SE3d curBrToW(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
        return curBrToW * _parMagr->EXTRI.SE3_CmToBr(topic);
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurEsToW(double timeByEs, const std::string &topic) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurEsToW' error, scale spline is not translation spline!!!");
    }
    double timeByBr = timeByEs + _parMagr->TEMPORAL.TO_EsToBr.at(topic);
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);

    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        Sophus::SE3d curBrToW(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
        return curBrToW * _parMagr->EXTRI.SE3_EsToBr(topic);
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurDnToW(double timeByDn, const std::string &topic) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurDnToW' error, scale spline is not translation spline!!!");
    }
    double timeByBr = timeByDn + _parMagr->TEMPORAL.TO_DnToBr.at(topic);
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);

    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        Sophus::SE3d curBrToW(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
        return curBrToW * _parMagr->EXTRI.SE3_DnToBr(topic);
    }
}

std::optional<Sophus::SE3d> CalibSolver::CurRjToW(double timeByRj, const std::string &topic) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        throw Status(Status::CRITICAL,
                     "'CurRjToW' error, scale spline is not translation spline!!!");
    }
    double timeByBr = timeByRj + _parMagr->TEMPORAL.TO_RjToBr.at(topic);
    const auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    const auto &posSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);

    if (!so3Spline.TimeStampInRange(timeByBr) || !posSpline.TimeStampInRange(timeByBr)) {
        return {};
    } else {
        Sophus::SE3d curBrToW(so3Spline.Evaluate(timeByBr), posSpline.Evaluate(timeByBr));
        return curBrToW * _parMagr->EXTRI.SE3_RjToBr(topic);
    }
}

TimeDeriv::ScaleSplineType CalibSolver::GetScaleType() {
    if (Configor::IsLiDARIntegrated() || Configor::IsPosCameraIntegrated()) {
        return TimeDeriv::ScaleSplineType::LIN_POS_SPLINE;
    } else if (Configor::IsRadarIntegrated() || Configor::IsRGBDIntegrated() ||
               Configor::IsVelCameraIntegrated() || Configor::IsEventIntegrated()) {
        return TimeDeriv::ScaleSplineType::LIN_VEL_SPLINE;
    } else {
        return TimeDeriv::ScaleSplineType::LIN_ACCE_SPLINE;
    }
}

CalibSolver::SplineBundleType::Ptr CalibSolver::CreateSplineBundle(double st,
                                                                   double et,
                                                                   double so3Dt,
                                                                   double scaleDt) {
    // create splines
    auto so3SplineInfo = ns_ctraj::SplineInfo(Configor::Preference::SO3_SPLINE,
                                              ns_ctraj::SplineType::So3Spline, st, et, so3Dt);
    auto scaleSplineInfo = ns_ctraj::SplineInfo(Configor::Preference::SCALE_SPLINE,
                                                ns_ctraj::SplineType::RdSpline, st, et, scaleDt);
    spdlog::info(
        "create spline bundle: start time: '{:.5f}', end time: '{:.5f}', so3 dt : '{:.5f}', scale "
        "dt: '{:.5f}'",
        st, et, so3Dt, scaleDt);
    return SplineBundleType::Create({so3SplineInfo, scaleSplineInfo});
}

void CalibSolver::AlignStatesToGravity() const {
    auto &so3Spline = _splines->GetSo3Spline(Configor::Preference::SO3_SPLINE);
    auto &scaleSpline = _splines->GetRdSpline(Configor::Preference::SCALE_SPLINE);
    // current gravity, velocities, and rotations are expressed in the reference frame
    // align them to the world frame whose negative z axis is aligned with the gravity vector
    auto SO3_RefToW =
        ObtainAlignedWtoRef(so3Spline.Evaluate(so3Spline.MinTime()), _parMagr->GRAVITY).inverse();
    _parMagr->GRAVITY = SO3_RefToW * _parMagr->GRAVITY;
    for (int i = 0; i < static_cast<int>(so3Spline.GetKnots().size()); ++i) {
        so3Spline.GetKnot(i) = SO3_RefToW * so3Spline.GetKnot(i);
    }
    for (int i = 0; i < static_cast<int>(scaleSpline.GetKnots().size()); ++i) {
        // attention: for three kinds of scale splines, this holds
        scaleSpline.GetKnot(i) = SO3_RefToW * scaleSpline.GetKnot(i) /* + Eigen::Vector3d::Zero()*/;
    }
}

void CalibSolver::StoreImagesForSfM(const std::string &topic,
                                    const std::set<IndexPair> &matchRes) const {
    // -------------
    // output images
    // -------------
    auto path = ns_ikalibr::Configor::DataStream::CreateImageStoreFolder(topic);
    if (path == std::nullopt) {
        throw ns_ikalibr::Status(Status::CRITICAL,
                                 "can not create path for image storing for topic: '{}'!!!", topic);
    }
    ImagesInfo info(topic, *path, {});
    const auto &frames = _dataMagr->GetCameraMeasurements(topic);

    int size = static_cast<int>(frames.size());
    const auto &intri = _parMagr->INTRI.Camera.at(topic);
    auto undistoMapper = VisualUndistortionMap::Create(intri);

    // downsample
    int64_t N = 1000;
    int64_t last_saved_time = -N;

    auto bar = std::make_shared<tqdm>();
    for (int i = 0; i != size; ++i) {
        bar->progress(i, size);
        const auto &frame = frames.at(i);
        int64_t frame_time = frame->GetId();
        if (frame_time - last_saved_time < N) continue;

        last_saved_time = frame_time;
        // generate the image name
        std::string filename = std::to_string(frame->GetId()) + ".jpg";
        info.images[frame->GetId()] = filename;

        cv::Mat undistImg = undistoMapper->RemoveDistortion(frame->GetImage());

        // save image
        cv::imwrite(*path + "/" + filename, undistImg);
    }
    bar->finish();

    // -------------------
    // colmap command line
    // -------------------
    auto ws = ns_ikalibr::Configor::DataStream::CreateSfMWorkspace(topic);
    if (ws == std::nullopt) {
        throw ns_ikalibr::Status(Status::CRITICAL,
                                 "can not create workspace for SfM for topic: '{}'!!!", topic);
    }
    const std::string database_path = *ws + "/database.db";
    const std::string &image_path = *path;
    const std::string match_list_path = *ws + "/matches.txt";
    const std::string &output_path = *ws;

    auto logger = spdlog::basic_logger_mt("sfm_cmd", *ws + "/sfm-command-line.txt", true);
    // feature extractor
    logger->info(
        "command line for 'feature_extractor' in colmap for topic '{}':\n"
        "colmap feature_extractor "
        "--database_path {} "
        "--image_path {} "
        "--ImageReader.camera_model PINHOLE "
        "--ImageReader.single_camera 1 "
        "--ImageReader.camera_params {:.3f},{:.3f},{:.3f},{:.3f}\n",
        topic, database_path, image_path, intri->FocalX(), intri->FocalY(),
        intri->PrincipalPoint()(0), intri->PrincipalPoint()(1));

    // feature match
    std::ofstream matchPairFile(match_list_path, std::ios::out);
    for (const auto &[view1Id, view2Id] : matchRes) {
        if (info.images.count(view1Id) > 0 && info.images.count(view2Id) > 0) {  
            matchPairFile << std::to_string(view1Id) + ".jpg ";
            matchPairFile << std::to_string(view2Id) + ".jpg" << std::endl;
        }
    }
    matchPairFile.close();

    logger->info(
        "command line for 'matches_importer' in colmap for topic '{}':\n"
        "colmap matches_importer "
        "--database_path {} "
        "--match_list_path {} "
        "--match_type pairs\n",
        topic, database_path, match_list_path);

    logger->info(
        "---------------------------------------------------------------------------------");
    logger->info(
        "- SfM Reconstruction in [COLMAP GUI | COLMAP MAPPER | GLOMAP MAPPER (RECOMMEND)]-");
    logger->info(
        "---------------------------------------------------------------------------------");
    logger->info("- Way 1: COLMAP GUI -");
    // reconstruction
    logger->info(
        "---------------------\n"
        "colmap gui "
        "--database_path {} "
        "--image_path {}",
        database_path, image_path, output_path);
    logger->info(
        "---------------------------------------------------------------------------------");
    logger->info("- Way 2: COLMAP MAPPER -");
    double init_max_error = IsRSCamera(topic) ? 2.0 : 1.0;
    // reconstruction
    logger->info(
        "------------------------\n"
        "colmap mapper "
        "--database_path {} "
        "--image_path {} "
        "--output_path {} "
        "--Mapper.init_min_tri_angle 25 "
        "--Mapper.init_max_error {} "
        "--Mapper.tri_min_angle 3 "
        "--Mapper.ba_refine_focal_length 0 "
        "--Mapper.ba_refine_principal_point 0",
        database_path, image_path, output_path, init_max_error);
    logger->info(
        "---------------------------------------------------------------------------------");
    logger->info("- Way 3: GLOMAP MAPPER (RECOMMEND) -");
    logger->info(
        "------------------------------------\n"
        "glomap mapper "
        "--database_path {} "
        "--image_path {} "
        "--output_path {}",
        database_path, image_path, output_path);
    logger->info(
        "---------------------------------------------------------------------------------\n");

    // format convert
    logger->info(
        "command line for 'model_converter' in colmap for topic '{}':\n"
        "colmap model_converter "
        "--input_path {} "
        "--output_path {} "
        "--output_type TXT\n",
        topic, output_path + "/0", output_path);
    logger->flush();
    spdlog::drop("sfm_cmd");

    std::ofstream file(ns_ikalibr::Configor::DataStream::GetImageStoreInfoFile(topic));
    auto ar = GetOutputArchiveVariant(file, Configor::Preference::OutputDataFormat);
    SerializeByOutputArchiveVariant(ar, Configor::Preference::OutputDataFormat,
                                    cereal::make_nvp("info", info));
}

ns_veta::Veta::Ptr CalibSolver::TryLoadSfMData(const std::string &topic,
                                               double errorThd,
                                               std::size_t trackLenThd) const {
    // info file
    const auto infoFilename = ns_ikalibr::Configor::DataStream::GetImageStoreInfoFile(topic);
    if (!std::filesystem::exists(infoFilename)) {
        spdlog::warn("the info file, i.e., '{}', dose not exists!!!", infoFilename);
        return nullptr;
    }

    const auto &sfmWsPath = ns_ikalibr::Configor::DataStream::CreateSfMWorkspace(topic);
    if (!sfmWsPath) {
        spdlog::warn("the sfm workspace for topic '{}' dose not exists!!!", topic);
        return nullptr;
    }

    const auto camerasFilename = *sfmWsPath + "/cameras.txt";
    if (!std::filesystem::exists(camerasFilename)) {
        spdlog::warn("the cameras file, i.e., '{}', dose not exists!!!", camerasFilename);
        return nullptr;
    }

    // images
    const auto imagesFilename = *sfmWsPath + "/images.txt";
    if (!std::filesystem::exists(imagesFilename)) {
        spdlog::warn("the images file, i.e., '{}', dose not exists!!!", imagesFilename);
        return nullptr;
    }

    // points
    const auto ptsFilename = *sfmWsPath + "/points3D.txt";
    if (!std::filesystem::exists(ptsFilename)) {
        spdlog::warn("the points 3D file, i.e., '{}', dose not exists!!!", ptsFilename);
        return nullptr;
    }

    // cameras

    // load info file
    ImagesInfo info("", "", {});
    {
        std::ifstream file(infoFilename);
        auto ar = GetInputArchiveVariant(file, Configor::Preference::OutputDataFormat);
        SerializeByInputArchiveVariant(ar, Configor::Preference::OutputDataFormat,
                                       cereal::make_nvp("info", info));
    }

    // load cameras
    auto cameras = ColMapDataIO::ReadCamerasText(camerasFilename);

    // load images
    auto images = ColMapDataIO::ReadImagesText(imagesFilename);

    // load landmarks
    auto points3D = ColMapDataIO::ReadPoints3DText(ptsFilename);

    auto veta = ns_veta::Veta::Create();

    // cameras
    assert(cameras.size() == 1);
    const auto &camera = cameras.cbegin()->second;
    assert(camera.params_.size() == 4);
    const auto &intriIdx = camera.camera_id_;
    auto intri = std::make_shared<ns_veta::PinholeIntrinsic>(*_parMagr->INTRI.Camera.at(topic));
    veta->intrinsics.insert({intriIdx, intri});

    // from images to our views and poses
    std::map<ns_veta::IndexT, CameraFrame::Ptr> ourIdxToCamFrame;
    for (const auto &frame : _dataMagr->GetCameraMeasurements(topic)) {
        ourIdxToCamFrame.insert({frame->GetId(), frame});
    }

    const auto &nameToOurIdx = info.GetImagesNameToIdx();
    for (const auto &[IdFromColmap, image] : images) {
        const auto &viewId = nameToOurIdx.at(image.name_);
        const auto &poseId = viewId;

        auto frameIter = ourIdxToCamFrame.find(viewId);
        // this frame is not involved in solving
        if (frameIter == ourIdxToCamFrame.cend()) {
            continue;
        }

        // view
        auto view = ns_veta::View::Create(
            // timestamp (aligned)
            frameIter->second->GetTimestamp(),
            // index
            viewId, intriIdx, poseId,
            // width, height
            intri->imgWidth, intri->imgHeight);
        veta->views.insert({viewId, view});

        // pose
        auto T_WorldToImg = ns_veta::Posed(image.QuatWorldToImg().matrix(), image.tvec_);
        // we store pose from camera to world
        veta->poses.insert({poseId, T_WorldToImg.Inverse()});
    }

    for (const auto &frame : _dataMagr->GetCameraMeasurements(topic)) {
        if (veta->views.count(frame->GetId()) == 0) {
            spdlog::warn(
                "frame indexed as '{}' of camera '{}' is involved in solving but not reconstructed "
                "in SfM!!!",
                frame->GetId(), topic);
        }
    }

    // from point3D to our structure
    for (const auto &[pt3dId, pt3d] : points3D) {
        // filter bad landmarks
        if (pt3d.error_ > errorThd || pt3d.track_.size() < trackLenThd) {
            continue;
        }

        auto &lm = veta->structure[pt3dId];
        lm.X = pt3d.xyz_;
        lm.color = pt3d.color_;

        for (const auto &track : pt3d.track_) {
            const auto &img = images.at(track.image_id);
            auto pt2d = img.points2D_.at(track.point2D_idx);

            if (pt3dId != pt2d.point3D_id_) {
                spdlog::warn(
                    "'point3D_id_' of point3D and 'point3D_id_' of feature connected are in "
                    "conflict!!!");
                continue;
            }

            const auto viewId = nameToOurIdx.at(img.name_);
            // this frame is not involved in solving
            if (veta->views.find(viewId) == veta->views.cend()) {
                continue;
            }

            lm.obs.insert({viewId, ns_veta::Observation(pt2d.xy_, track.point2D_idx)});
        }
        if (lm.obs.size() < trackLenThd) {
            veta->structure.erase(pt3dId);
        }
    }

    return veta;
}

void CalibSolver::PerformTransformForVeta(const ns_veta::Veta::Ptr &veta,
                                          const ns_veta::Posed &curToNew,
                                          double scale) {
    // pose
    for (auto &[id, pose] : veta->poses) {
        pose.Translation() *= scale;
        pose = curToNew * pose;
    }

    // structure
    for (auto &[id, lm] : veta->structure) {
        lm.X *= scale;
        lm.X = curToNew(lm.X);
    }
}

bool CalibSolver::IsRSCamera(const std::string &topic) {
    CameraModelType type = CameraModelType::GS;
    if (auto iterCam = Configor::DataStream::CameraTopics.find(topic);
        iterCam != Configor::DataStream::CameraTopics.cend()) {
        type = EnumCast::stringToEnum<CameraModelType>(iterCam->second.Type);
    } else if (auto iterRGBD = Configor::DataStream::RGBDTopics.find(topic);
               iterRGBD != Configor::DataStream::RGBDTopics.cend()) {
        type = EnumCast::stringToEnum<CameraModelType>(iterRGBD->second.Type);
    } else if (auto iterEvent = Configor::DataStream::EventTopics.find(topic);
               iterEvent != Configor::DataStream::EventTopics.cend()) {
        type = EnumCast::stringToEnum<CameraModelType>(iterEvent->second.Type);
    }
    return IsOptionWith(CameraModelType::RS, type);
}

void CalibSolver::DownsampleVeta(const ns_veta::Veta::Ptr &veta,
                                 std::size_t lmNumThd,
                                 std::size_t obvNumThd) {
    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    if (veta->structure.size() > lmNumThd) {
        std::vector<ns_veta::IndexT> lmIdVec;
        lmIdVec.reserve(veta->structure.size());
        for (const auto &[lmId, lm] : veta->structure) {
            lmIdVec.push_back(lmId);
        }
        auto lmIdVecToMove =
            SamplingWoutReplace2(engine, lmIdVec, veta->structure.size() - lmNumThd);
        for (const auto &id : lmIdVecToMove) {
            veta->structure.erase(id);
        }
    }
    for (auto &[lmId, lm] : veta->structure) {
        if (lm.obs.size() < obvNumThd) {
            continue;
        }

        std::vector<ns_veta::IndexT> obvIdVec;
        obvIdVec.reserve(lm.obs.size());
        for (const auto &[viewId, obv] : lm.obs) {
            obvIdVec.push_back(viewId);
        }

        auto obvIdVecToMove = SamplingWoutReplace2(engine, obvIdVec, lm.obs.size() - obvNumThd);
        for (const auto &id : obvIdVecToMove) {
            lm.obs.erase(id);
        }
    }
}

void CalibSolver::SaveStageCalibParam(const CalibParamManager::Ptr &par, const std::string &desc) {
    const static std::string paramDir = Configor::DataStream::OutputPath + "/iteration/stage";
    if (!std::filesystem::exists(paramDir) && !std::filesystem::create_directories(paramDir)) {
        spdlog::warn("create directory failed: '{}'", paramDir);
    } else {
        const std::string paramFilename =
            paramDir + "/" + desc + ns_ikalibr::Configor::GetFormatExtension();
        par->Save(paramFilename, ns_ikalibr::Configor::Preference::OutputDataFormat);
    }
}

ns_veta::Veta::Ptr CalibSolver::CreateVetaFromOpticalFlow(
    const std::string &topic,
    const std::vector<OpticalFlowCorr::Ptr> &traceVec,
    const ns_veta::PinholeIntrinsic::Ptr &intri,
    const std::function<std::optional<Sophus::SE3d>(
        const CalibSolver *, double, const std::string &)> &SE3_CurSenToW) const {
    if (GetScaleType() != TimeDeriv::LIN_POS_SPLINE) {
        return nullptr;
    }
    auto veta = std::make_shared<ns_veta::Veta>();

    // intrinsics
    const ns_veta::IndexT INTRI_ID = 0;
    veta->intrinsics.insert({INTRI_ID, intri});

    ns_veta::IndexT LM_ID_COUNTER = 0;
    for (const auto &corr : traceVec) {
        if (corr->depth < 1E-3 /* 1 mm */) {
            continue;
        }
        auto SE3_CurDnToW = SE3_CurSenToW(this, corr->frame->GetTimestamp(), topic);
        if (SE3_CurDnToW == std::nullopt) {
            continue;
        }

        // index
        ns_veta::IndexT viewId = corr->frame->GetId(), intriIdx = INTRI_ID, poseId = viewId;

        // we store pose from camera to world
        auto pose = ns_veta::Posed(SE3_CurDnToW->so3(), SE3_CurDnToW->translation());

        // view
        auto view = ns_veta::View::Create(
            // timestamp (aligned)
            corr->frame->GetTimestamp(),
            // index
            viewId, intriIdx, poseId,
            // width, height
            intri->imgWidth, intri->imgHeight);

        // landmark
        const double depth = corr->depth;
        Eigen::Vector2d lmInDnPlane = intri->ImgToCam(corr->MidPoint());
        Eigen::Vector3d lmInDn(lmInDnPlane(0) * depth, lmInDnPlane(1) * depth, depth);
        Eigen::Vector3d lmInW = *SE3_CurDnToW * lmInDn;
        // this landmark has only one observation
        auto lm = ns_veta::Landmark(lmInW, {{viewId, ns_veta::Observation(corr->MidPoint(), 0)}});
        auto cImg = corr->frame->GetColorImage();
        // b, g, r
        auto color = cImg.at<cv::Vec3b>((int)corr->MidPoint()(1), (int)corr->MidPoint()(0));
        // r, g, b <- r, g, b
        lm.color = {color(2), color(1), color(0)};

        veta->poses.insert({poseId, pose});
        veta->views.insert({viewId, view});
        veta->structure.insert({++LM_ID_COUNTER, lm});
    }

    return veta;
}

void CalibSolver::AddGyroFactor(Estimator::Ptr &estimator,
                                const std::string &imuTopic,
                                Estimator::Opt option) const {
    double weight = Configor::DataStream::IMUTopics.at(imuTopic).GyroWeight;

    for (const auto &item : _dataMagr->GetIMUMeasurements(imuTopic)) {
        estimator->AddIMUGyroMeasurement(item, imuTopic, option, weight);
    }
}

std::vector<Eigen::Vector2d> CalibSolver::FindTexturePoints(const cv::Mat &eventFrame, int num) {
    cv::Mat imgFiltered;
    cv::medianBlur(eventFrame, imgFiltered, 5);

    cv::Mat gray;
    cv::cvtColor(imgFiltered, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, num, 0.01 /*qualityLevel*/, 10 /*minDistance*/);

    std::vector<Eigen::Vector2d> vertex;
    vertex.reserve(corners.size());

    for (const auto &corner : corners) {
        vertex.emplace_back(corner.x, corner.y);
        // DrawKeypointOnCVMat(imgFiltered, corner, true, cv::Scalar(0, 0, 0));
    }

    cv::imshow("Edges", edges);
    cv::imshow("Filtered Event Frame", imgFiltered);
    // cv::waitKey(0);

    return vertex;
}

std::vector<Eigen::Vector2d> CalibSolver::FindTexturePointsAt(
    const std::vector<EventArrayPtr>::const_iterator &tarIter,
    const std::vector<EventArrayPtr> &data,
    std::size_t eventNumThd,
    const ns_veta::PinholeIntrinsic::Ptr &intri,
    int featNum) {
    std::size_t accumulatedEventNum = 0;
    auto fIter = tarIter, bIter = tarIter;

    while (true) {
        bool updated = false;
        if (std::distance(data.cbegin(), fIter) > 0) {
            accumulatedEventNum += (*fIter)->GetEvents().size();
            if (accumulatedEventNum > eventNumThd) {
                break;
            } else {
                fIter = std::prev(fIter);
            }
            updated = true;
        }

        if (std::distance(bIter, data.cend()) > 0) {
            accumulatedEventNum += (*bIter)->GetEvents().size();
            if (accumulatedEventNum > eventNumThd) {
                break;
            } else {
                bIter = std::next(bIter);
            }
            updated = true;
        }
        if (!updated) {
            break;
        }
    }
    auto mat = EventArray::DrawRawEventFrame(fIter, bIter, intri);
    auto vertex = FindTexturePoints(EventArray::DrawRawEventFrame(fIter, bIter, intri), featNum);
    for (const auto &v : vertex) {
        DrawKeypointOnCVMat(mat, v, true, cv::Scalar(0, 0, 0));
    }
    cv::imshow("Event-Based Feature Tracking Initial Points", mat);
    cv::waitKey(0);
    return vertex;
}

std::vector<Eigen::Vector2d> CalibSolver::GenUniformSeeds(const ns_veta::PinholeIntrinsicPtr &intri,
                                                          int seedNum,
                                                          int padding) {
    auto w = intri->imgWidth, h = intri->imgHeight;
    int delta = static_cast<int>(
        std::sqrt(static_cast<double>((w - 2 * padding) * (h - 2 * padding)) / seedNum));
    std::vector<Eigen::Vector2d> seeds;
    for (int x = padding; x < static_cast<int>(w) - padding; x += delta) {
        for (int y = padding; y < static_cast<int>(h) - padding; y += delta) {
            seeds.emplace_back(x, y);
        }
    }
    return seeds;
}

void CalibSolver::SaveEventDataForFeatureTracking(const std::string &topic,
                                                  const std::string &ws,
                                                  double BATCH_TIME_WIN_THD,
                                                  std::size_t seedNum) const {
    const auto &intri = _parMagr->INTRI.Camera.at(topic);
    const auto &eventMes = _dataMagr->GetEventMeasurements(topic);
    auto saeCreator = ActiveEventSurface::Create(intri, 0.01);

    std::stringstream commands;
    // the index of sub batch in event data
    int subEventDataIdx = 0;
    std::vector<EventsInfo::SubBatch> subBatches;

    auto headIter = eventMes.cbegin();
    for (auto tailIter = eventMes.cbegin(); tailIter != eventMes.cend(); ++tailIter) {
        /**
         *                          |<- BATCH_TIME_WIN_THD ->|
         * ------------------------------------------------------------------
         * |<- BATCH_TIME_WIN_THD ->|                        |<- BATCH_TIME_WIN_THD ->|
         *  data in windown would be output for event-based feature tracking
         */
        if ((*tailIter)->GetTimestamp() - (*headIter)->GetTimestamp() < BATCH_TIME_WIN_THD) {
            continue;
        }

        // information for seed
        decltype(tailIter) seedIter;
        cv::Mat tsMatSeedTime;
        // treated as reset of 'accumEventMat'
        cv::Mat accumEventMat = saeCreator->GetEventImgMat(true, false);
        bool findSeed = false;
        std::size_t accumulatedEventNum = 0;
        for (auto iter = headIter; iter != tailIter; ++iter) {
            saeCreator->GrabEvent(*iter, true);
            accumulatedEventNum += (*iter)->GetEvents().size();
            /**
             *        |--> event data to be accumulated to locate seed positions
             * ----|-------------------|----
             *     |<--batch windown-->|
             *        |--> the seed time
             */
            if (!findSeed && (*iter)->GetTimestamp() - (*headIter)->GetTimestamp() > 0.01) {
                // assign
                seedIter = iter;
                findSeed = true;
                // create time surface
                tsMatSeedTime = saeCreator->TimeSurface(true,   // ignore polarity
                                                        false,  // undisto event frame mat
                                                        0,      // perform medianBlur
                                                        0.02);  // the constant decay rate
                accumEventMat = saeCreator->GetEventImgMat(true, false);
            }
        }

        if (!findSeed) {
            // update
            headIter = tailIter;
            continue;
        }

        // find seeds (todo: refine)
        std::vector<cv::Point2f> ptsCurVec;
        cv::goodFeaturesToTrack(tsMatSeedTime, ptsCurVec, seedNum, 0.01, 10);

        cv::cvtColor(tsMatSeedTime, tsMatSeedTime, cv::COLOR_GRAY2BGR);
        std::vector<Eigen::Vector2d> seeds;
        seeds.reserve(ptsCurVec.size());
        for (const auto &pt : ptsCurVec) {
            DrawKeypointOnCVMat(tsMatSeedTime, pt, true);
            DrawKeypointOnCVMat(accumEventMat, pt, true);
            seeds.push_back({pt.x, pt.y});
        }

        // the directory to save sub event data
        const std::string subWS = ws + "/" + std::to_string(subEventDataIdx);
        if (!std::filesystem::exists(subWS)) {
            if (!std::filesystem::create_directories(subWS)) {
                throw Status(Status::CRITICAL,
                             "can not create sub output directory '{}' for event "
                             "camera '{}' sub event data sequence '{}'!!!",
                             subWS, topic, subEventDataIdx);
            }
        }
        double seedTime = (*seedIter)->GetEvents().back()->GetTimestamp();
        auto [c, i] = HASTEDataIO::SaveRawEventDataAsBinary(headIter,  // from
                                                            tailIter,  // to
                                                            intri,     // intrinsics
                                                            seeds,     // seed positions
                                                            seedTime,  // seed timestamps
                                                            subWS,     // directory
                                                            subEventDataIdx);

        // record information
        commands << '\"' << c << "\"\n";
        subBatches.emplace_back(i);

        // plus index of sub data batch
        ++subEventDataIdx;

        cv::Mat m;
        cv::hconcat(tsMatSeedTime, accumEventMat, m);
        cv::imshow("Normalized Time Surface & Accumulated Event Mat", m);
        cv::waitKey(0);

        // update
        headIter = tailIter;
    }

    // the command shell file
    const std::string cmdOutputPath = ws + "/run_haste.sh";
    std::ofstream ofCmdShell(cmdOutputPath, std::ios::out);
    ofCmdShell << "#!/bin/bash\n"
                  "commands=(\n"
               << commands.str()
               << ")\n"
                  "max_parallel=8\n"
                  "echo \"Maximum Parallel Tasks Set To: $max_parallel\"\n"
                  "total_commands=${#commands[@]}\n"
                  "completed_commands=0\n"
                  "update_progress() {\n"
                  "  ((completed_commands++))\n"
                  "  percentage=$((completed_commands * 100 / total_commands))\n"
                  "  echo \"Currently Completed HASTE-Based Event Feature Tracking Tasks: "
                  "[$completed_commands / $total_commands]-[$percentage%]\"\n"
                  "}\n"
                  "for cmd in \"${commands[@]}\"; do\n"
                  "  current_cmd=\"$cmd\"\n"
                  "  $cmd > /dev/null 2>&1 &\n"
                  "  running=$(jobs -r | wc -l)\n"
                  "  if [ \"$running\" -ge \"$max_parallel\" ]; then\n"
                  "    wait -n\n"
                  "    update_progress\n"
                  "  fi\n"
                  "done\n"
                  "while [ $(jobs -r | wc -l) -gt 0 ]; do\n"
                  "  wait -n\n"
                  "  update_progress\n"
                  "done\n"
                  "echo -e \"\\nAll Commands Completed!\"\n";
    ofCmdShell.close();

    EventsInfo info(topic, ws, _dataMagr->GetRawStartTimestamp(), subBatches);
    HASTEDataIO::SaveEventsInfo(info, ws);
#if 0

        /**
         * |--> 'outputSIter1'
         * |            |<- BATCH_TIME_WIN_THD ->|<- BATCH_TIME_WIN_THD ->|
         * ------------------------------------------------------------------
         * |<- BATCH_TIME_WIN_THD ->|<- BATCH_TIME_WIN_THD ->|
         * | data in this windown would be output for event-based feature tracking
         * |--> 'outputSIter2'
         */
        const double BATCH_TIME_WIN_THD_HALF = BATCH_TIME_WIN_THD * 0.5;
        const auto &intri = _parMagr->INTRI.Camera.at(topic);

        const auto &eventMes = _dataMagr->GetEventMeasurements(topic);
        // this queue maintain two iterators
        std::queue<std::vector<EventArray::Ptr>::const_iterator> batchHeadIter;
        batchHeadIter.push(eventMes.cbegin());
        batchHeadIter.push(eventMes.cbegin());

        // only for visualization
        auto matSIter = eventMes.cbegin();
        std::size_t accumulatedEventCount = 0;
        cv::Mat eventFrameMat;

        std::vector<EventsInfo::SubBatch> subBatches;
        std::stringstream commands;

        // the index of sub batch in event data
        int subEventDataIdx = 0;

        // for visualization
        std::shared_ptr<tqdm> bar = std::make_shared<tqdm>();
        auto totalSubBatchCount =
            (eventMes.back()->GetTimestamp() - eventMes.front()->GetTimestamp()) /
            BATCH_TIME_WIN_THD * 2;

        for (auto curIter = matSIter; curIter != eventMes.cend(); ++curIter) {
            accumulatedEventCount += (*curIter)->GetEvents().size();
            if (accumulatedEventCount > EVENT_FRAME_NUM_THD) {
                /**
                 * If the number of events accumulates to a certain number, we construct it into an
                 * event frame. The event frame is used for visualization, and for calculating rough
                 * texture positions for feature tracking (haste-based). Note that this even frame
                 * is a distorted one
                 */
                // eventFrameMat = EventArray::DrawRawEventFrame(matSIter, curIter, intri);
                // cv::imshow("Event Frame", eventFrameMat);
                // cv::waitKey(1);

                matSIter = curIter;
                accumulatedEventCount = 0;
            }

            auto headIter = batchHeadIter.front(), tailIter = batchHeadIter.back();

            if ((*curIter)->GetTimestamp() - (*tailIter)->GetTimestamp() >
                BATCH_TIME_WIN_THD_HALF) {
                if ((*curIter)->GetTimestamp() - (*headIter)->GetTimestamp() > BATCH_TIME_WIN_THD) {
                    bar->progress(subEventDataIdx, static_cast<int>(totalSubBatchCount));

                    // the directory to save sub event data
                    const std::string subWS = ws + "/" + std::to_string(subEventDataIdx);
                    if (!std::filesystem::exists(subWS)) {
                        if (!std::filesystem::create_directories(subWS)) {
                            throw Status(Status::CRITICAL,
                                         "can not create sub output directory '{}' for event "
                                         "camera '{}' sub event data sequence '{}'!!!",
                                         subWS, topic, subEventDataIdx);
                        }
                    }

                    /**
                     *        |--> event data to be accumulated to locate seed positions
                     * ----|-------------------|----
                     *     |<--batch windown-->|
                     *     |--> the seed time
                     */
                    // calculate rough texture positions for feature tracking (haste-based)
                    auto seedIter = headIter;
                    /**
                     * feature points are extracted as seeds
                     */
                    while ((*seedIter)->GetTimestamp() - (*headIter)->GetTimestamp() < 0.01) {
                        ++seedIter;
                    }
                    auto seeds = FindTexturePointsAt(seedIter,  // the reference iterator
                                                     eventMes, EVENT_FRAME_NUM_THD, intri, seedNum);

                    /**
                     * another way to select seeds is direct uniform selection
                     */
                    // auto seeds = GenUniformSeeds(intri, 200 /*generate about 200 seeds*/, 10);

                    const double seedsTime = (*seedIter)->GetTimestamp();

                    auto [command, batchInfo] =
                        HASTEDataIO::SaveRawEventDataAsBinary(headIter,   // from
                                                              curIter,    // to
                                                              intri,      // intrinsics
                                                              seeds,      // seed positions
                                                              seedsTime,  // seed timestamps
                                                              subWS,      // directory
                                                              subEventDataIdx);

                    // record information
                    commands << '\"' << command << "\"\n";
                    subBatches.emplace_back(batchInfo);

                    // plus index of sub data batch
                    ++subEventDataIdx;
                }

                batchHeadIter.push(curIter);
                batchHeadIter.pop();
            }
        }

        bar->progress(subEventDataIdx, subEventDataIdx);
        bar->finish();

        // the command shell file
        const std::string cmdOutputPath = ws + "/run_haste.sh";
        std::ofstream ofCmdShell(cmdOutputPath, std::ios::out);
        ofCmdShell << "#!/bin/bash\n"
                      "commands=(\n"
                   << commands.str()
                   << ")\n"
                      "max_parallel=8\n"
                      "echo \"Maximum Parallel Tasks Set To: $max_parallel\"\n"
                      "total_commands=${#commands[@]}\n"
                      "completed_commands=0\n"
                      "update_progress() {\n"
                      "  ((completed_commands++))\n"
                      "  percentage=$((completed_commands * 100 / total_commands))\n"
                      "  echo \"Currently Completed HASTE-Based Event Feature Tracking Tasks: "
                      "[$completed_commands / $total_commands]-[$percentage%]\"\n"
                      "}\n"
                      "for cmd in \"${commands[@]}\"; do\n"
                      "  current_cmd=\"$cmd\"\n"
                      "  $cmd > /dev/null 2>&1 &\n"
                      "  running=$(jobs -r | wc -l)\n"
                      "  if [ \"$running\" -ge \"$max_parallel\" ]; then\n"
                      "    wait -n\n"
                      "    update_progress\n"
                      "  fi\n"
                      "done\n"
                      "while [ $(jobs -r | wc -l) -gt 0 ]; do\n"
                      "  wait -n\n"
                      "  update_progress\n"
                      "done\n"
                      "echo -e \"\\nAll Commands Completed!\"\n";
        ofCmdShell.close();

        EventsInfo info(topic, ws, _dataMagr->GetRawStartTimestamp(), subBatches);
        HASTEDataIO::SaveEventsInfo(info, ws);
#endif
}
}  // namespace ns_ikalibr