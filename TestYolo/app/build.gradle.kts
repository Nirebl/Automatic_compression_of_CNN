plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.testyolo"
    compileSdk = 36 // можно 35/34, если у тебя такой SDK установлен

    defaultConfig {
        applicationId = "com.example.testyolo"
        // РЕКОМЕНДУЮ: minSdk = 24, чтобы приложение ставилось на большее число устройств
        minSdk = 26
        targetSdk = 36

        versionCode = 1
        versionName = "1.0"

        ndk {
            // Kotlin DSL: добавляем через +=
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                // Ставим API для CMake равным minSdk (исправит android-26 в вызове)
                arguments += listOf("-DANDROID_PLATFORM=android-${minSdk}")
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    // В AGP/KTS не используем aaptOptions внутри defaultConfig.
    // Нужное поведение дает androidResources.noCompress:
    androidResources {
        noCompress += listOf("param", "bin")
    }

    buildFeatures {
        prefab = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // Kotlin DSL: используем val, а не def
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("com.google.android.material:material:1.12.0")

}
