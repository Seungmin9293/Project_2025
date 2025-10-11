import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screen/info_input_screen.dart';
import 'screen/recruitment_screen.dart';

void main() => runApp(const SeniorRecruitApp());

class SeniorRecruitApp extends StatelessWidget {
  const SeniorRecruitApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Senior Recruit',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final double maxButtonWidth =
    MediaQuery.of(context).size.width.clamp(0, 730); // 최대 730px
    return Scaffold(
      backgroundColor: const Color(0xFFEFF3F8), // 연한 그레이‑블루
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(vertical: 40),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(
                  'Senior Recruit',
                  style: GoogleFonts.pacifico(
                    fontSize: 64,
                    color: const Color(0xFF1E3D8F),
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                const Text(
                  '시니어를 위한 맞춤형 채용 서비스',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF546E7A),
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 4),
                const Text(
                  '경험과 지혜를 인정받는 새로운 시작',
                  style: TextStyle(
                    fontSize: 18,
                    color: Color(0xFF90A4AE),
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 60),

                // ── 버튼 섹션 ─────────────────────────────────────────
                GradientButton(
                  width: maxButtonWidth,
                  gradient: const LinearGradient(
                    colors: [Color(0xFF2E71FF), Color(0xFF1749FF)],
                  ),
                  leadingIcon: Icons.person_add_alt_1,
                  title: '정보입력',
                  subtitle: '개인정보 및 경력사항을 입력해주세요',
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const InfoInputScreen(),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 24),
                GradientButton(
                  width: maxButtonWidth,
                  gradient: const LinearGradient(
                    colors: [Color(0xFF02AE6E), Color(0xFF019260)],
                  ),
                  leadingIcon: Icons.search,
                  title: '채용추천받기',
                  subtitle: '맞춤형 일자리를 추천받아보세요',
                  onTap: () { Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => const JobRecommend(),
                    ),
                  );
                  },
                ),
                const SizedBox(height: 24),
                GradientButton(
                  width: maxButtonWidth,
                  gradient: const LinearGradient(
                    colors: [Color(0xFFF7A613), Color(0xFFE28400)],
                  ),
                  leadingIcon: Icons.phone,
                  title: '채용전화걸기',
                  subtitle: '직접 상담원과 통화하여 도움받기',
                  onTap: () {},
                ),

                // ── 푸터 안내 ──────────────────────────────────────
                const SizedBox(height: 60),
                const Text(
                  '도움이 필요하시면 언제든지 문의해주세요',
                  style: TextStyle(fontSize: 16, color: Color(0xFF90A4AE)),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    Icon(Icons.phone, size: 20, color: Color(0xFF90A4AE)),
                    SizedBox(width: 4),
                    Text('----------',
                        style:
                        TextStyle(fontSize: 16, color: Color(0xFF90A4AE))),
                    SizedBox(width: 24),
                    Icon(Icons.access_time,
                        size: 20, color: Color(0xFF90A4AE)),
                    SizedBox(width: 4),
                    Text('평일 09:00‑18:00',
                        style:
                        TextStyle(fontSize: 16, color: Color(0xFF90A4AE))),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class GradientButton extends StatelessWidget {
  const GradientButton({
    super.key,
    required this.width,
    required this.gradient,
    required this.leadingIcon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final double width;
  final Gradient gradient;
  final IconData leadingIcon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    const double height = 120;

    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: onTap,
      child: Container(
        width: width,
        height: height,
        padding: const EdgeInsets.symmetric(horizontal: 28),
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            // 좌측 아이콘 (연한 색의 동그란 배경)
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.25),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(leadingIcon, color: Colors.white, size: 32),
            ),
            const SizedBox(width: 24),
            // 타이틀 + 서브타이틀
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      fontSize: 16,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
            // 우측 화살표
            const Icon(Icons.arrow_forward, color: Colors.white, size: 32),
          ],
        ),
      ),
    );
  }
}