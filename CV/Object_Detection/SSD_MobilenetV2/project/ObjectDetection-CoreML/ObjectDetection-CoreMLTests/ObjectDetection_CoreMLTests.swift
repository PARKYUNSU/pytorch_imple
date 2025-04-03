//
//  ObjectDetection_CoreMLTests.swift
//  SSDMobileNet-CoreMLTests
//
//  Created by GwakDoyoung on 01/02/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import XCTest
import Vision

class ObjectDetection_CoreMLTests: XCTestCase {

    // MARK: - Vision Properties
    let image = UIImage(named: "IMG_8564")
    var pixelBuffer: CVPixelBuffer?
    
    override func setUp() {
        // image configuration
        pixelBuffer = image?.pixelBufferFromImage()
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    // MobileNetV2_SSDLite 모델을 테스트하는 함수로 변경
    func testPerformanceMobileNetV2_SSDLiteModel() {
        // MobileNetV2_SSDLite 모델로 변경
        let model: VNCoreMLModel? = try? VNCoreMLModel(for: MobileNetV2_SSDLite().model)
        let ssdRequest: VNCoreMLRequest? = VNCoreMLRequest(model: model!, completionHandler: nil)
        ssdRequest?.imageCropAndScaleOption = .scaleFill
        
        guard let pixelBuffer = pixelBuffer,
            let request = ssdRequest else {
                fatalError()
        }
        
        self.measure {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            try? handler.perform([request])
        }
    }

}
