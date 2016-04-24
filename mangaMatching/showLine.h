#pragma once

#include "mangaShow.h"
mangaShow ms;

namespace mangaMatching {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;

	/// <summary>
	/// showLine 的摘要
	/// </summary>
	public ref class showLine : public System::Windows::Forms::Form
	{
	public:
		showLine(void)
		{
			InitializeComponent();
			//
			//TODO:  在此加入建構函式程式碼
			//
		}

	protected:
		/// <summary>
		/// 清除任何使用中的資源。
		/// </summary>
		~showLine()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::PictureBox^  showLine_pictureBox;
	private: System::Windows::Forms::GroupBox^  lineCheck_groupBox;
	private: System::Windows::Forms::Label^  image_label;
	private: System::Windows::Forms::Label^  graph_label;
	private: System::Windows::Forms::Label^  curve_label;
	private: System::Windows::Forms::PictureBox^  sampleFace_pictureBox;


	protected:

	private:
		/// <summary>
		/// 設計工具所需的變數。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器
		/// 修改這個方法的內容。
		/// </summary>
		void InitializeComponent(void)
		{
			this->showLine_pictureBox = (gcnew System::Windows::Forms::PictureBox());
			this->lineCheck_groupBox = (gcnew System::Windows::Forms::GroupBox());
			this->image_label = (gcnew System::Windows::Forms::Label());
			this->graph_label = (gcnew System::Windows::Forms::Label());
			this->curve_label = (gcnew System::Windows::Forms::Label());
			this->sampleFace_pictureBox = (gcnew System::Windows::Forms::PictureBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->showLine_pictureBox))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->sampleFace_pictureBox))->BeginInit();
			this->SuspendLayout();
			// 
			// showLine_pictureBox
			// 
			this->showLine_pictureBox->AllowDrop = true;
			this->showLine_pictureBox->BackColor = System::Drawing::Color::Black;
			this->showLine_pictureBox->Location = System::Drawing::Point(0, 0);
			this->showLine_pictureBox->Name = L"showLine_pictureBox";
			this->showLine_pictureBox->Size = System::Drawing::Size(894, 861);
			this->showLine_pictureBox->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->showLine_pictureBox->TabIndex = 0;
			this->showLine_pictureBox->TabStop = false;
			this->showLine_pictureBox->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &showLine::showLine_pictureBox_DragDrop);
			this->showLine_pictureBox->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &showLine::showLine_pictureBox_DragEnter);
			// 
			// lineCheck_groupBox
			// 
			this->lineCheck_groupBox->Location = System::Drawing::Point(900, 114);
			this->lineCheck_groupBox->Name = L"lineCheck_groupBox";
			this->lineCheck_groupBox->Size = System::Drawing::Size(284, 747);
			this->lineCheck_groupBox->TabIndex = 1;
			this->lineCheck_groupBox->TabStop = false;
			// 
			// image_label
			// 
			this->image_label->BackColor = System::Drawing::Color::Gray;
			this->image_label->Font = (gcnew System::Drawing::Font(L"微軟正黑體", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(136)));
			this->image_label->ForeColor = System::Drawing::Color::White;
			this->image_label->Location = System::Drawing::Point(920, 13);
			this->image_label->Name = L"image_label";
			this->image_label->Size = System::Drawing::Size(252, 24);
			this->image_label->TabIndex = 2;
			this->image_label->Text = L"Image";
			this->image_label->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// graph_label
			// 
			this->graph_label->BackColor = System::Drawing::Color::Gray;
			this->graph_label->Font = (gcnew System::Drawing::Font(L"微軟正黑體", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(136)));
			this->graph_label->ForeColor = System::Drawing::Color::White;
			this->graph_label->Location = System::Drawing::Point(920, 43);
			this->graph_label->Name = L"graph_label";
			this->graph_label->Size = System::Drawing::Size(252, 24);
			this->graph_label->TabIndex = 2;
			this->graph_label->Text = L"Graph";
			this->graph_label->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// curve_label
			// 
			this->curve_label->BackColor = System::Drawing::Color::Gray;
			this->curve_label->Font = (gcnew System::Drawing::Font(L"微軟正黑體", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(136)));
			this->curve_label->ForeColor = System::Drawing::Color::White;
			this->curve_label->Location = System::Drawing::Point(920, 73);
			this->curve_label->Name = L"curve_label";
			this->curve_label->Size = System::Drawing::Size(252, 24);
			this->curve_label->TabIndex = 2;
			this->curve_label->Text = L"Curve";
			this->curve_label->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// sampleFace_pictureBox
			// 
			this->sampleFace_pictureBox->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(64)),
				static_cast<System::Int32>(static_cast<System::Byte>(64)), static_cast<System::Int32>(static_cast<System::Byte>(64)));
			this->sampleFace_pictureBox->Location = System::Drawing::Point(668, 639);
			this->sampleFace_pictureBox->Name = L"sampleFace_pictureBox";
			this->sampleFace_pictureBox->Size = System::Drawing::Size(226, 222);
			this->sampleFace_pictureBox->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->sampleFace_pictureBox->TabIndex = 3;
			this->sampleFace_pictureBox->TabStop = false;
			// 
			// showLine
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 12);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1184, 861);
			this->Controls->Add(this->sampleFace_pictureBox);
			this->Controls->Add(this->curve_label);
			this->Controls->Add(this->graph_label);
			this->Controls->Add(this->image_label);
			this->Controls->Add(this->lineCheck_groupBox);
			this->Controls->Add(this->showLine_pictureBox);
			this->Name = L"showLine";
			this->Text = L"showLine";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->showLine_pictureBox))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->sampleFace_pictureBox))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void showLine_pictureBox_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
				 if (e->Data->GetDataPresent(DataFormats::FileDrop)){
					 array<System::String^>^files = (array<System::String^>^)e->Data->GetData(DataFormats::FileDrop);
					 System::String ^imgExt = ".jpg|.png|.bmp|.jpeg|.gif";
					 System::String ^graphExt = ".graph";
					 System::String ^curveExt = ".curve";
					 try{
						 for (int i = 0; i < files->Length; ++i){
							 System::String ^ext = Path::GetExtension(files[i]);
							 if (imgExt->IndexOf(ext) >= 0){
								 char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(files[i]).ToPointer();
								 ms.read_img(fileName);
								 this->image_label->BackColor = System::Drawing::Color::Green;
							 }
							 else if (graphExt->IndexOf(ext) >= 0){
								 char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(files[i]).ToPointer();
								 ms.read_graph(fileName, MANGA_FACE);
								 this->graph_label->BackColor = System::Drawing::Color::Green;
							 }
							 else if (curveExt->IndexOf(ext) >= 0){
								 char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(files[i]).ToPointer();
								 ms.read_graph(fileName, SAMPLE_FACE);
								 this->curve_label->BackColor = System::Drawing::Color::Green;
								 //ms.calculate_curve();
							 }
						 }
						 if (ms.is_read_img() && ms.is_read_mangaFace()){
							 ms.draw_curves(false);
							 if (ms.is_read_sampleFace()){
								 //ms.find_seed();
								 //this->sampleFace_pictureBox->Image = ms.get_sample_canvas_Bitmap();
								 ms.test();
							 }
							 this->showLine_pictureBox->Image = ms.get_canvas_Bitmap();

							 this->lineCheck_groupBox->Controls->Clear();
							 std::vector<bool> curves_drawable = ms.get_curves_drawable();
							 for (int i = 0; i < curves_drawable.size(); ++i){
								 CheckBox ^cb = gcnew System::Windows::Forms::CheckBox();
								 cb->AutoSize = true;
								 cb->Size = System::Drawing::Size(77, 16);
								 cb->Location = System::Drawing::Point(7 + cb->Size.Width * (i % 3), i / 3 * cb->Size.Height + 16);
								 cb->Name = L"line" + (i > 9 ? "" : "0") + i;
								 cb->Text = L"line" + (i > 9 ? "" : "0") + i;
								 cb->TabIndex = i + 2;
								 cb->Checked = curves_drawable[i];
								 cb->UseVisualStyleBackColor = true;
								 cb->CheckStateChanged += gcnew System::EventHandler(this, &showLine::cb_CheckStateChanged);
								 this->lineCheck_groupBox->Controls->Add(cb);
							 }
						 }
					 }
					 catch (System::Exception^ ex){
						 MessageBox::Show(ex->Message);
						 return;
					 }
				 }
	}
	private: System::Void showLine_pictureBox_DragEnter(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
				 if (e->Data->GetDataPresent(DataFormats::FileDrop))
					 e->Effect = DragDropEffects::Copy;
				 else
					 e->Effect = DragDropEffects::None;
	}
	private: System::Void cb_CheckStateChanged(System::Object^  sender, System::EventArgs^  e) {
				 std::vector<bool> curves_drawable = ms.get_curves_drawable();
				 for (int i = 0; i < curves_drawable.size(); ++i){
					 ms.set_curves_drawable(i, ((CheckBox ^)this->lineCheck_groupBox->Controls[i])->Checked);
				 }
				 ms.draw_curves();
				 this->showLine_pictureBox->Image = ms.get_canvas_Bitmap();

	}
	};
}
